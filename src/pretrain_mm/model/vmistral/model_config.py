""" VMistral model configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "HuggingFaceM4/VLM_WebSight_finetuned": "https://huggingface.co/HuggingFaceM4/VLM_WebSight_finetuned/resolve/main/config.json",
}


class VMistralVisionConfig(PretrainedConfig):
    r""" """
    model_type = "vmistral"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        _flash_attn_2_enabled=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self._flash_attn_2_enabled = _flash_attn_2_enabled


class VMistralPerceiverConfig(PretrainedConfig):
    r"""
    TThis is the configuration class to store the configuration of a [`MistralModel`]. It is used to instantiate an
    Mistral model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mistral-7B-v0.1 or Mistral-7B-Instruct-v0.1.

    [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
    [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        use_resampler (`bool`, *optional*, defaults to `False`):
            Whether or not to use the resampler
        resampler_n_latents (`int`, *optional*, defaults to ):
            Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).
        resampler_depth (`int`, *optional*, defaults to 6):
            Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (< 3).
        resampler_n_heads (`int`, *optional*, defaults to 16):
            Number of heads in each Transformer block (for multi-headed self-attention).
        resampler_head_dim (`int`, *optional*, defaults to 96):
            Dimensionality of each head projection in the Transformer block.
        qk_layer_norms_perceiver (`bool`, *optional*, defaults to `False`):
            Whether or not to use qk layer norms in perceiver
    """
    model_type = "vmistral"

    def __init__(
        self,
        resampler_n_latents=64,
        resampler_depth=6,
        resampler_n_heads=16,
        resampler_head_dim=96,
        qk_layer_norms_perceiver=False,
        **kwargs,
    ):
        self.resampler_n_latents = resampler_n_latents
        self.resampler_depth = resampler_depth
        self.resampler_n_heads = resampler_n_heads
        self.resampler_head_dim = resampler_head_dim
        self.qk_layer_norms_perceiver = qk_layer_norms_perceiver

        super().__init__(**kwargs)


class VMistralConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MistralModel`]. It is used to instantiate an
    Mistral model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mistral-7B-v0.1 or Mistral-7B-Instruct-v0.1.

    [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
    [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        additional_vocab_size (`int`, *optional`, defaults to 0):
            Additional vocabulary size of the model, typically for the special "<img>" token. Additional vocab tokens
            are always trainable whereas regular vocab tokens can be frozen or not.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Mistral model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MistralModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
            The maximum sequence length that this model might ever be used with. Mistral's sliding window attention
            allows sequence of up to 4096*32 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        alpha_initializer (`str`, *optional*, defaults to `"zeros"`):
            Initialization type for the alphas.
        alphas_initializer_range (`float`, *optional*, defaults to 0.0):
            The standard deviation of the truncated_normal_initializer for initializing the alphas in the Gated Cross
            Attention.
        alpha_type (`str`, *optional*, defaults to `"float"`):
            Whether the gating alphas should be vectors or single floats.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention window size. If not specified, will default to `4096`.
        cross_layer_interval (`int`, *optional*, default to 1)
            Interval for cross attention (from text to image) layers.
        qk_layer_norms (`bool`, *optional*, defaults to `False`): Whether to add layer norm after q and k
        freeze_text_layers (`bool`, *optional*, defaults to `True`): Whether to freeze text layers
        freeze_text_module_exceptions (`bool`, *optional*, defaults to `[]`):
            Exceptions to freezing text layers when `freeze_text_layers` is `True`
        freeze_lm_head (`bool`, *optional*, defaults to `False`): Whether to freeze lm head
        freeze_vision_layers (`bool`, *optional*, defaults to `True`):  Whether to freeze vision layers
        freeze_vision_module_exceptions (`bool`, *optional*, defaults to `[]`):
            Exceptions to freezing vision layers when `freeze_vision_layers` is `True`
        use_resampler (`bool`, *optional*, defaults to `False`): Whether to use the Resampler
        vision_config (`IdeficsVisionConfig`,  *optional*): Custom vision config or dict
        perceiver_config (`IdeficsPerceiverConfig`,  *optional*): Custom perceiver config or dict

    Example:
    ```python
    >>> from transformers import MistralModel, MistralConfig

    >>> # Initializing a Mistral 7B style configuration
    >>> configuration = MistralConfig()

    >>> # Initializing a model from the Mistral 7B style configuration
    >>> model = MistralModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "vmistral"
    is_composition = False

    def __init__(
        self,
        additional_vocab_size=0,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        alpha_initializer="zeros",
        alphas_initializer_range=0.0,
        alpha_type="float",
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,  # None in the original configuration_mistral, we set it to the unk_token_id
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=32_001,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        cross_layer_interval=1,
        qk_layer_norms=False,
        freeze_text_layers=True,
        freeze_text_module_exceptions=[],
        freeze_lm_head=False,
        freeze_vision_layers=True,
        freeze_vision_module_exceptions=[],
        attention_dropout=0.0,
        _flash_attn_2_enabled=True,
        use_resampler=False,
        vision_config=None,
        perceiver_config=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.additional_vocab_size = additional_vocab_size
        self.image_token_id = image_token_id
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.alpha_initializer = alpha_initializer
        self.alphas_initializer_range = alphas_initializer_range
        self.alpha_type = alpha_type
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        self.cross_layer_interval = cross_layer_interval
        self.qk_layer_norms = qk_layer_norms
        self.freeze_vision_layers = freeze_vision_layers

        self.freeze_text_layers = freeze_text_layers
        self.freeze_text_module_exceptions = freeze_text_module_exceptions
        self.freeze_vision_module_exceptions = freeze_vision_module_exceptions
        self.freeze_lm_head = freeze_lm_head

        self.use_resampler = use_resampler
        self._flash_attn_2_enabled = _flash_attn_2_enabled
        self.attention_dropout = attention_dropout

        if perceiver_config is None:
            self.perceiver_config = VMistralPerceiverConfig()
        elif isinstance(perceiver_config, dict):
            self.perceiver_config = VMistralPerceiverConfig(**perceiver_config)
        elif isinstance(perceiver_config, VMistralPerceiverConfig):
            self.perceiver_config = perceiver_config

        if vision_config is None:
            self.vision_config = VMistralVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = VMistralVisionConfig(**vision_config)
        elif isinstance(vision_config, VMistralVisionConfig):
            self.vision_config = vision_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # IMPORTANT: Do not do any __init__ args-based checks in the constructor, since
        # PretrainedConfig.from_dict first instantiates the class with the config dict and only then
        # updates the config object with `kwargs` from from_pretrained, so during the instantiation
        # of this object many attributes have default values and haven't yet been overridden.
        # Do any required checks inside `from_pretrained` once the superclass' `from_pretrained` was run.
