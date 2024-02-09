from typing import Any, TypeAlias

import easyocr
import numpy as np
import paddleocr
import pytesseract

# paddleocr returns like (list[Result]) where Result is [Points, (Text, Probability)] and Points are [int, int]:
# [[[[2.0, 3.0], [90.0, 3.0], [90.0, 18.0], [2.0, 18.0]], ('Search Reddit', 0.9748642444610596)]]
# easyocr returns like (list[Result]) where Result is (Points, Text, Probability) and Points are List[int, int]:
# [[[[2, 3], [90, 3], [90, 18], [2, 18]], 'Search Reddit', 0.9748642444610596]]

PaddleOCRResult: TypeAlias = list[list[list[list[float]] | tuple[str, float]]]
EasyOCRResult: TypeAlias = list[list[list[list[int]] | str | float]]


TesseractGroupByResult: TypeAlias = "pd.GroupBy"

init_kwargs_paddleocr_defaults = {
    "use_angle_cls": True,
    "lang": "en",
    "use_gpu": True,
    "show_log": False,
    "use_mp": True,
}

init_kwargs_easyocr_defaults = {
    "lang_list": ["en"],
    "gpu": True,
}

# otherwise text can be float if the text is numbers
kwargs_pytesseract = {
    "output_type": "data.frame",
    "pandas_config": {
        "dtype": {"text": str},
    },
}


def make_get_type(fn: callable) -> callable:
    def get_type_from_result(result: PaddleOCRResult | EasyOCRResult | None):
        if result is None:
            return []

        return [fn(val) for val in result]

    return get_type_from_result


# Tesseract output data.frame is DataFrame with columns:
#    ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text']
def get_groupedby(df: "pd.DataFrame") -> TesseractGroupByResult:
    return df[df.conf != -1].groupby(["page_num", "block_num", "par_num", "line_num"])


def get_text_tesseract(groupby_result: TesseractGroupByResult):
    return groupby_result.text.apply(lambda x: " ".join(list(x))).tolist()


def get_probability_tesseract(groupby_result: TesseractGroupByResult):
    # conf score from tesseract is out of 100, the other two are 0-1
    # mean is for words on same block/page/paragraph/line
    return groupby_result.conf.mean().apply(lambda x: x / 100).tolist()


# Note: Was using these to get each val but i think zip is better/quicker as not going through the list twice
get_text_paddleocr = make_get_type(lambda x: x[1][0])
get_probability_paddleocr = make_get_type(lambda x: x[1][1])

get_text_easyocr = make_get_type(lambda x: x[1])
get_probability_easyocr = make_get_type(lambda x: x[2])


class MultiOCRLabeler:
    def __init__(
        self,
        use_easy: bool | easyocr.Reader = True,
        use_paddle: bool | paddleocr.PaddleOCR = True,
        use_tesseract: bool = True,
        kwargs_paddle: dict = init_kwargs_paddleocr_defaults,
        kwargs_easy: dict = init_kwargs_easyocr_defaults,
    ):
        """tool to label images with 3rd party OCR.
        Allow the use_[ocrtool] because paddleocr is not serializable so it doesnt cache with datasets.map and doesnt work with datasets.map in general
        this also means that it can be infeasible to use paddleocr if it needs to be serialized in some manner (like mp or datasets.map)
        tried tons of different ways to pickle it but it just doesnt work and the github repo is all chinese.

        i think easyocr has the same issue with gpu.

        Args:
            use_easyocr (bool, optional): _description_. Defaults to True.
            use_paddle (bool, optional): _description_. Defaults to True.
            use_tesseract (bool, optional): _description_. Defaults to True.
            init_kwargs_paddleocr (dict, optional): _description_. Defaults to init_kwargs_paddleocr.
        """

        self.paddleocr = paddleocr.PaddleOCR(**kwargs_paddle) if (use_paddle == True) else use_paddle
        self.easyocr = easyocr.Reader(**kwargs_easy) if (use_easy == True) else use_easy

        self.use_tesseract = use_tesseract

    def __call__(self, image: "Image") -> dict[str, dict[str, list]]:
        image_arr = np.asarray(image)

        prob_results = {}
        text_results = {}

        if self.easyocr != False:
            easy_result = self._easyocr_ocr(image_arr)
            _, easy_texts, easy_probs = zip(*easy_result) if easy_result else ([], [], [])
            prob_results["easy"], text_results["easy"] = list(easy_probs), list(easy_texts)

        if self.paddleocr != False:
            paddle_result = self._paddleocr_ocr(image_arr)[0]
            paddle_texts, paddle_probs = zip(*[pair[1] for pair in paddle_result]) if paddle_result else ([], [])
            prob_results["paddle"], text_results["paddle"] = list(paddle_probs), list(paddle_texts)

        if self.use_tesseract != False:
            tess_df = self._tesseract_ocr(image_arr)
            tess_grouped = tess_df[tess_df.conf != -1].groupby(["page_num", "block_num", "par_num", "line_num"])
            prob_results["tesseract"], text_results["tesseract"] = get_probability_tesseract(
                tess_grouped
            ), get_text_tesseract(tess_grouped)

        return {
            "text": text_results,
            "prob": prob_results,
        }

    def _paddleocr_ocr(self, image):
        result = self.paddleocr.ocr(image, cls=True)
        return result

    def _easyocr_ocr(self, image):
        result = self.easyocr.readtext(image)
        return result

    def _tesseract_ocr(self, image):
        result = pytesseract.image_to_data(image, **kwargs_pytesseract)
        return result


class OCRLabeler:
    def __init__(self, init_kwargs: dict = {}, ocr_init_func: callable = None):
        self.init_kwargs = init_kwargs
        self.ocr_init_func = ocr_init_func

        # self.reset()

    def __call__(self, image: "Image", **kwargs) -> Any:
        return self.ocr_func(image, **kwargs)

    def reset(self, **kwargs):
        self.init_kwargs = {**self.init_kwargs, **kwargs}
        self.ocr_func = self.ocr_init_func(**self.init_kwargs)


class TesseractLabeler(OCRLabeler):
    def __init__(self):
        # super().__init__(ocr_init_func=pytesseract.image_to_data, init_kwargs=kwargs_pytesseract)
        self.ocr_func = pytesseract.image_to_data
        self.ocr_kwargs = kwargs_pytesseract

    def __call__(self, image: "Image", **kwargs) -> dict[str, list]:
        # grouped = get_groupedby(super().__call__(image, **kwargs))
        ocr_kwargs = {**self.ocr_kwargs, **kwargs}
        grouped = get_groupedby(self.ocr_func(image, **ocr_kwargs))
        return {
            "text": get_text_tesseract(grouped),
            "prob": get_probability_tesseract(grouped),
        }


class PaddleOCRLabeler(OCRLabeler):
    def __init__(self, use_gpu: bool = True):
        self.init_kwargs = {**init_kwargs_paddleocr_defaults, "use_gpu": use_gpu}

        self.ocr_inst = paddleocr.PaddleOCR(**self.init_kwargs)
        self.ocr_func = self.ocr_inst.ocr

    def __call__(self, image: "Image", **kwargs) -> dict[str, list]:
        result = super().__call__(image, **kwargs)
        return {
            "text": get_text_paddleocr(result),
            "prob": get_probability_paddleocr(result),
        }


if __name__ == "__main__":

    from PIL import Image

    image = Image.open("output/tmp_current_image.png")
    cropped_image = Image.open("output/cropped-image.png")
    small_cropped_image = Image.open("output/ez-testcrop.png")
    labeler = OCRLabeler()

    # tes_result = comparer.tesseract_ocr(small_cropped_image)
    # tes_result = pytesseract.image_to_string(small_cropped_image)
    result = labeler(small_cropped_image)
    breakpoint()
