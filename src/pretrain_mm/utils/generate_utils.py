import torch
import torch.nn.functional as F

from pretrain_mm import logger


def sample_single(logits, temperature, top_k: int = None):
    # pluck the logits at the final step and scale by desired temperature
    logits = logits[:, -1, :] / temperature
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")
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
    processor: callable,
    inputs: dict,
    max_new_tokens: int = 10,
    stop_tokens: list[int] = [],
    temperature: float = 1.0,
    top_k: int = None,
    indices_placeholder: torch.Tensor = torch.tensor([[-1]]),
    mask_placeholder: torch.Tensor = torch.tensor([[1]]),
    drop_last_of_input: bool = True,
    disable_progress_bar: bool = True,
):
    # switch devices for placeholders
    indices_placeholder = indices_placeholder.to(model.device)
    mask_placeholder = mask_placeholder.to(model.device)

    model_inputs = processor(**inputs).to(model.device)
    image_patches_indices = model_inputs.image_patches_indices
    image_patches = model_inputs.image_patches
    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask

    if drop_last_of_input:
        # think i need to chop off last bit as processor is wrong
        image_patches_indices = image_patches_indices[:, :-1]
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

    progress = logger.progress(start=True, ensure_exit=True, disable=disable_progress_bar)
    ptask = progress.add_task(f"[cyan]Generating: ", total=max_new_tokens)

    for _ in range(max_new_tokens):
        model_output = model(
            input_ids=input_ids,
            image_patches=image_patches,
            image_patches_indices=image_patches_indices,
            attention_mask=attention_mask,
        )
        progress.update(ptask, advance=1)

        idx_next = sample_single(model_output.logits, temperature=temperature, top_k=top_k)

        input_ids = torch.cat([input_ids, idx_next], dim=-1)
        image_patches_indices = torch.cat([image_patches_indices, indices_placeholder], dim=-1)
        attention_mask = torch.cat([attention_mask, mask_placeholder], dim=-1)

        if idx_next in stop_tokens:
            # logger.info(f"found stop token: {idx_next[0, 0].item()}")
            break
    progress.stop()
    return input_ids
