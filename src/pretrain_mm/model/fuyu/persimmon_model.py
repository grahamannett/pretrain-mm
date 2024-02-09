import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.persimmon.modeling_persimmon import (
    PersimmonAttention,
    PersimmonDecoderLayer,
    apply_rotary_pos_emb,
)
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

from pretrain_mm import logger

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# should mimick FlashAttention2 from Llama2
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L390
class PersimmonFlashAttention2(PersimmonAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_value: Cache = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]:

        bsz, q_len, _ = hidden_states.size()
        fused_qkv = self.query_key_value(hidden_states)

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_states, key_states, value_states) = self._split_heads(fused_qkv)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, num_heads, head_dim]
        query_states = query_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_emb.dim],
            query_states[..., self.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_emb.dim],
            key_states[..., self.rotary_emb.dim :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        if past_key_value is not None:
            # Specific to RoPE models with partial rotation
            cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_emb.dim}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # -- END OF NORMAL PERSIMMON ATTENTION
        dropout_rate = self.config.attention_dropout if self.training else 0.0

        # Inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim].
        # We would need to refactor the KV cache to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.query_key_value.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # [batch_size, seq_length, num_heads, head_dim] for all
        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        causal = self.is_causal if not self._flash_attn_uses_top_left_mask else self.is_causal and query_length != 1
        # Contains at least one padding token in the sequence
        # if attention_mask is not None:
        #     batch_size = query_states.shape[0]
        #     breakpoint()
        #     query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
        #         query_states, key_states, value_states, attention_mask, query_length
        #     )

        #     cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        #     max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        #     attn_output_unpad = flash_attn_varlen_func(
        #         query_states,
        #         key_states,
        #         value_states,
        #         cu_seqlens_q=cu_seqlens_q,
        #         cu_seqlens_k=cu_seqlens_k,
        #         max_seqlen_q=max_seqlen_in_batch_q,
        #         max_seqlen_k=max_seqlen_in_batch_k,
        #         dropout_p=dropout,
        #         softmax_scale=softmax_scale,
        #         causal=causal,
        #     )

        #     attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        # else:
        #     attn_output = flash_attn_func(
        #         query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
        #     )

        # -- NOTE: NOT USING ATTENTION MASK UNTIL I HAVE ACCESS TO GPUS WITH FLASH-ATTN
        # -- NOTE: without patching forward of PersimmonModel cant use attention mask i dont think
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
        )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


def patch_persimmon_decoder_layer(cls):
    original_init = cls.__init__

    def patched_init(self, config, layer_idx: int, *args, **kwargs):
        logger.warning_once("Monkey-patched PersimmonDecoderLayer for flash-attn. If there are issues, disable this.")
        original_init(self, config, layer_idx, *args, **kwargs)
        self.self_attn = PersimmonFlashAttention2(config=config, layer_idx=layer_idx)

    cls.__init__ = patched_init
    return cls


patch_persimmon_decoder_layer(PersimmonDecoderLayer)


if __name__ == "__main__":
    import transformers

    inp = torch.arange(10)[None, ...]
    attention_mask = torch.ones_like(inp, dtype=torch.bool)
    attention_mask[0, 0] = False

    pmodel = transformers.PersimmonForCausalLM.from_pretrained(
        "adept/persimmon-8b-base",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # output = pmodel(inp.to(pmodel.device))
    output = pmodel(inp.to(pmodel.device), attention_mask=attention_mask.to(pmodel.device))
