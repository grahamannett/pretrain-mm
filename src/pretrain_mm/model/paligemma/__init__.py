from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from pretrain_mm.model.fuyu.fuyu_constants import FuyuConstantsClass


class PaliGemmaConstantsClass(FuyuConstantsClass):
    pass


PaliGemmaConstants = PaliGemmaConstantsClass()

MODEL_ID: str = "google/paligemma-3b-ft-docvqa-896"
