import torch.nn.functional as F


def make_loss_func(model, ignore_index: int = -100):
    # moved to here for time being as using the inbuilt loss function on the model is more ideal
    # since that allows for FSDP to be used more simply, versus this might now work
    def get_loss(shift_logits, shift_labels):
        #
        # Shift so that tokens < n predict n + Flatten the tokens
        # shift_logits = shift_logits[..., :-1, :].contiguous()
        # shift_labels = shift_labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)
        return loss

    return get_loss


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
