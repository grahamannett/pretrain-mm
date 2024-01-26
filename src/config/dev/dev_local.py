CommonScreensDatasetInfo = {
    "image_dir": "/Users/graham/code/clippymm_datasets/pretrain_data/common-screens/s3/data/jpeg",
    "header_path": "/Users/graham/code/clippymm_datasets/pretrain_data/common-screens/s3/metadata/common-screens-with-meta-2022-header.txt",
    "metadata_path": "/Users/graham/code/clippymm_datasets/pretrain_data/common-screens/s3/metadata/common-screens-with-meta-2022-12.csv",
}

Mind2WebDatasetInfo = {
    "train": {
        "dataset_path": "osunlp/Mind2Web",
    },
    "test": {
        "dataset_path": "osunlp/Mind2Web",  # just reusing the train dataset for now
    },
}




def _default_shorten_model(model: "torch.nn.Module", num_layers: int = 2) -> "torch.nn.Module":
    import transformers

    get_layers = {
        "FuyuForCausalLM": lambda model: model.language_model.model.layers,
    }

    if isinstance(model, transformers.models.fuyu.modeling_fuyu.FuyuForCausalLM):
        model.language_model.model.layers = model.language_model.model.layers[:num_layers]
    if isinstance(model, transformers.models.mistral.modeling_mistral.MistralForCausalLM):
        model.model.layers = model.model.layers[:num_layers]
    return model


def setup_model_for_dev(
    model: "torch.nn.Module", num_layers: int = 4, transform_model_func: callable = _default_shorten_model
) -> "torch.nn.Module":
    model = transform_model_func(model, num_layers)
    return model
