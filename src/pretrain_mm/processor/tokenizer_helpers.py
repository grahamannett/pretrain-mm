from transformers import PreTrainedTokenizer, PreTrainedModel


def update_with_actions(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, new_tokens: list[str]):
    _ = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
