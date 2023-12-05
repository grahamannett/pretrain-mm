import torch
import torch.nn.functional as F


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
