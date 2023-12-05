def get_embeddings(model, input_ids, image_patches, image_patches_indices, **kwargs):
    # should be similar to forward of model
    # ideally would just hook into model but register_forward_hook doesnt seem to include
    # the inputs for hf models
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    patch_embeddings = [
        model.vision_embed_tokens(patch.to(model.vision_embed_tokens.weight.dtype)).squeeze(0)
        for patch in image_patches
    ]
    input_embeds = model.gather_continuous_embeddings(
        word_embeddings=inputs_embeds,
        continuous_embeddings=patch_embeddings,
        image_patch_input_indices=image_patches_indices,
    )
    return input_embeds
