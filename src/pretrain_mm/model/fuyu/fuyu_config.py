from transformers import FuyuConfig as BaseFuyuConfig


class FuyuConfig(BaseFuyuConfig):
    def __init__(
        self,
        causal_lm_loss: dict | None = None,
        patch_image_out: bool = False,
        patch_idx_latent: bool = False,
        patch_gather_continuous_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.patch_image_out = patch_image_out
        self.patch_idx_latent = patch_idx_latent
        self.patch_gather_continuous_embeddings = patch_gather_continuous_embeddings
        self.causal_lm_loss = causal_lm_loss

    def patch(self, num_hidden_layers: int = 0):
        # passing these in via from_pretrained does not work unless you also pass the text_config dict
        # as it will set num_hidden_layers but not on the text config.  HF library is broken/a bit weird
        if num_hidden_layers > 0:
            self.num_hidden_layers = num_hidden_layers
            self.text_config.num_hidden_layers = num_hidden_layers
        return self
