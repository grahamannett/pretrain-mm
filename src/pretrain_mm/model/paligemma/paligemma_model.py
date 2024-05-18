from transformers import PaliGemmaForConditionalGeneration as HFPaliGemmaForConditionalGeneration
from transformers import PaliGemmaProcessor as HFPaliGemmaProcessor

from pretrain_mm.model.fuyu.fuyu_constants import FuyuConstantsClass
from pretrain_mm.processor.processor import ProcessorMixin


class PaliGemmaForConditionalGeneration(HFPaliGemmaForConditionalGeneration):
    pass

class PaliGemmaProcessor(HFPaliGemmaProcessor, ProcessorMixin):
    pass


class PaliGemmaConstantsClass(FuyuConstantsClass):
    pass


PaliGemmaConstants = PaliGemmaConstantsClass()

MODEL_ID: str = "google/paligemma-3b-ft-docvqa-896"
