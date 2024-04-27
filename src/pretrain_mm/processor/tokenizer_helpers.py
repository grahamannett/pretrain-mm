from transformers import PreTrainedModel, PreTrainedTokenizer


def update_with_actions(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, new_tokens: list[str]):
    if (tokens_added := tokenizer.add_tokens(new_tokens)) != len(new_tokens):
        raise ValueError("tokens_added != len(new_tokens)")

    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer
