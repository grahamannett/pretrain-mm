from transformers import AutoTokenizer, AutoProcessor

text_only_tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    model_max_length=512,
    padding_side="left",
    add_eos_token=True,
)

mm_processor = AutoProcessor.from_pretrained(
    "adept/fuyu-8b",
    model_max_length=4096,
)
