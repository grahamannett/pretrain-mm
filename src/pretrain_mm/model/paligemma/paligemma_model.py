from transformers import PaliGemmaForConditionalGeneration as HFPaliGemmaForConditionalGeneration
from transformers import PaliGemmaProcessor as HFPaliGemmaProcessor
from transformers.utils import TensorType

from pretrain_mm.model.fuyu.fuyu_constants import FuyuConstantsClass
from pretrain_mm.processor.processor import ProcessorMixin


class PaliGemmaForConditionalGeneration(HFPaliGemmaForConditionalGeneration):
    pass


class PaliGemmaProcessor(HFPaliGemmaProcessor, ProcessorMixin):
    def __call__(
        self,
        text=None,
        images=None,
        tokenize_newline_separately=True,
        padding=False,
        truncation=None,
        max_length=None,
        return_tensors=TensorType.PYTORCH,
        do_resize=None,
        do_normalize=None,
        image_mean=None,
        image_std=None,
        data_format="channels_first",
        input_data_format=None,
        resample=None,
        do_convert_rgb=None,
        do_thumbnail=None,
        do_align_long_axis=None,
        do_rescale=None,
        **kwargs,
    ):
        return super().__call__(
            text=text,
            images=images,
            tokenize_newline_separately=tokenize_newline_separately,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            do_resize=do_resize,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            data_format=data_format,
            input_data_format=input_data_format,
            resample=resample,
            do_convert_rgb=do_convert_rgb,
            do_thumbnail=do_thumbnail,
            do_align_long_axis=do_align_long_axis,
            do_rescale=do_rescale,
        )


class PaliGemmaConstantsClass(FuyuConstantsClass):
    pass


PaliGemmaConstants = PaliGemmaConstantsClass()

MODEL_ID: str = "google/paligemma-3b-ft-docvqa-896"
