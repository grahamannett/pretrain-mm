def fuyu_post_processor(sample):
    sample.image_patches = [patch.squeeze(0) for patch in sample.image_patches]
    sample.image_patches_indices = sample.image_patches_indices.squeeze(0)

    sample.input_ids = sample.input_ids.squeeze(0)
    sample.attention_mask = sample.attention_mask.squeeze(0)
    return sample
