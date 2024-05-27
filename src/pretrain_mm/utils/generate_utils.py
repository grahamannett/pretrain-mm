import torch
import torch.nn.functional as F

from pretrain_mm.constants import NEG_INF


class StopOnToken:
    def __init__(self, stop_tokens: list[int]):
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] > 1:
            raise NotImplementedError("only handling batch size of 1")

        if input_ids[:, -1] in self.stop_tokens:
            return True
        return False

    def __repr__(self):
        return f"StopOnToken({self.stop_tokens})"


def sample_with_constrainer(logits, constrainer: callable, tok_idx: int, **kwargs):
    next_idx = constrainer(logits, tok_idx=tok_idx, **kwargs)
    return next_idx


def sample_single(logits, temperature, top_k: int = None, force_ids_mask: torch.Tensor = None, **kwargs):
    # pluck the logits at the final step and scale by desired temperature
    logits = logits[:, -1, :] / temperature
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = NEG_INF

    if force_ids_mask is not None:
        # mask out uses 1s to indicate force words so that we can * if wanted to mask out as well
        logits[..., ~force_ids_mask] = NEG_INF

    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution
    idx_next = torch.multinomial(probs, num_samples=1)
    return idx_next


def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
    max_context_length: int = 16_000,
    stop_tokens: list[int] = [],
):
    """
    seems like something is causing model.generate to hang, perhaps this will fix

    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= max_context_length else idx[:, -max_context_length:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)

        idx_next = sample_single(logits, temperate=temperature, top_k=top_k)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

        if idx_next in stop_tokens:
            break

    return idx


@torch.no_grad
def generate_helper(
    model: torch.nn.Module,
    processor: callable = None,
    inputs: dict = None,
    model_inputs: dict = None,
    max_new_tokens: int = 10,
    stop_ids: list[int] = [],
    force_words_ids: list[int] = [],
    force_ids_mask: torch.Tensor = None,
    temperature: float = 1.0,
    top_k: int = None,
    indices_placeholder: torch.Tensor = torch.tensor([[-1]]),
    mask_placeholder: torch.Tensor = torch.tensor([[1]]),
    drop_last_of_input: bool = False,  # this is only necessary if we are using old processor
    constrainer: callable = None,
    return_extra: callable = False,  # if return extra then wont return only tokens
    forward_kwargs: dict = {},  # for model.forward to allow hidden states etc
    use_past_key_values: bool = False,
) -> dict:
    # assert return_only_tokens ^ (any((return_last_logits, return_masked_logits))), "If return..."

    sample_func = sample_single if constrainer is None else sample_with_constrainer

    # switch devices for placeholders
    indices_placeholder = indices_placeholder.to(model.device)
    mask_placeholder = mask_placeholder.to(model.device)

    if model_inputs is None:
        model_inputs = processor(**inputs).to(model.device)

    image_patches_indices = model_inputs["image_patches_indices"]
    image_patches = model_inputs["image_patches"]
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    if drop_last_of_input:
        # necessary when using old processor as it adds extra tokens
        image_patches_indices = image_patches_indices[..., :-1]
        input_ids = input_ids[..., :-1]
        attention_mask = attention_mask[..., :-1]

    def _get_model_output(_input_ids, _image_patches, _image_patches_indices, _attention_mask):
        return model(
            input_ids=_input_ids,
            image_patches=_image_patches,
            image_patches_indices=_image_patches_indices,
            attention_mask=_attention_mask,
            **forward_kwargs,
        )

    # get single output no matter what, useful if we just want the logits for sequence for some eval related stuff
    model_output = _get_model_output(input_ids, image_patches, image_patches_indices, attention_mask)
    device = model_output.logits.device

    for tok_idx in range(0, max_new_tokens):
        if force_words_ids and (force_ids_mask is None):
            # make mask as 0s so that we can multiply to mask as well
            force_ids_mask = torch.zeros(model_output.logits.shape[-1], dtype=torch.bool, device=device)
            force_ids_mask[force_words_ids] = True

        idx_next = sample_func(
            logits=model_output.logits,
            temperature=temperature,
            top_k=top_k,
            tok_idx=tok_idx,
            constrainer=constrainer,
            # mask later or if you matmul then do it in here so you can keep the original logits
            force_ids_mask=force_ids_mask,
        )

        input_ids = torch.cat([input_ids, idx_next], dim=-1)
        image_patches_indices = torch.cat([image_patches_indices, indices_placeholder], dim=-1)
        attention_mask = torch.cat([attention_mask, mask_placeholder], dim=-1)

        if idx_next in stop_ids:
            break

        if use_past_key_values:
            model_output = model(
                input_ids=input_ids[:, -1:], past_key_values=model_output.past_key_values, **forward_kwargs
            )
        else:
            model_output = _get_model_output(input_ids, image_patches, image_patches_indices, attention_mask)

    if not return_extra:
        return {"input_ids": input_ids}  # just return the tokens generated

    _return_fn = return_extra if callable(return_extra) else _return_fn_default
    return _return_fn(model_output, input_ids)


def _return_fn_default(mod_out, inp, **kwargs):
    other = {}

    if hs := getattr(mod_out, "hidden_states", None):
        # bs x num_layers x seq_len x hidden_dim
        other["hidden_states"] = torch.stack(hs, dim=1).cpu()

    return {
        "input_ids": inp.detach().cpu(),
        "logits": mod_out.logits.detach().cpu(),
        **other,
    }
