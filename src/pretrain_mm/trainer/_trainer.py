def train_step(model, batch, loss_func):
    batch.to(model.device)
    input_ids = batch.input_ids
    attention_mask = batch.attention_mask
    image_patches = batch.image_patches
    image_patches_indices = batch.image_patches_indices

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_patches=image_patches,
        image_patches_indices=image_patches_indices,
    )

    loss = loss_func(outputs.logits, input_ids)

    return loss
