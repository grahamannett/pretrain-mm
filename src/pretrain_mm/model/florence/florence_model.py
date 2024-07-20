"""
florence info

https://huggingface.co/microsoft/Florence-2-large-ft/blob/main/processing_florence2.py

## model variants
- https://huggingface.co/microsoft/Florence-2-base



jupyter notebook:
- https://huggingface.co/microsoft/Florence-2-large/blob/main/modeling_florence2.py
- https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb

"""

from _modeling_florence import Florence2VisionModelWithProjection, Florence2LanguageForConditionalGeneration

MODEL_ID: str = "microsoft/Florence-2-large"

# class Model

model = Florence2LanguageForConditionalGeneration.from_pretrained(MODEL_ID)

