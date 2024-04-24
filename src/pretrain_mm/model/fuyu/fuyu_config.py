from transformers import FuyuConfig as BaseFuyuConfig


class FuyuConfig(BaseFuyuConfig):
    def __init__(
        self,
        patch_image_out: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_image_out = patch_image_out

    def patch(self, num_hidden_layers: int = 0):
        # passing these in via from_pretrained does not work.  it only passes to the ???
        # as it will set num_hidden_layers but not on the text config.  HF library is so broken
        if num_hidden_layers > 0:
            self.num_hidden_layers = num_hidden_layers
            self.text_config.num_hidden_layers = num_hidden_layers
