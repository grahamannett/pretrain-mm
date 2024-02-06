from typing import TypeAlias

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


class OCRLabeler:
    def __init__(
        self,
        use_easy: bool | easyocr.Reader = True,
        use_paddle: bool | paddleocr.PaddleOCR = True,
        use_tesseract: bool = True,
        init_kwargs_paddleocr: dict = init_kwargs_paddleocr_defaults,
        init_kwargs_easyocr: dict = init_kwargs_easyocr_defaults,
        _use_gpu: bool = None,
        _paddleinst=None,
        _easyinst=None,
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
        # if isinstance(_use_gpu, bool):
        #     init_kwargs_paddleocr["use_gpu"] = _use_gpu
        #     init_kwargs_easyocr["gpu"] = _use_gpu

        self.paddleocr = paddleocr.PaddleOCR(**init_kwargs_paddleocr) if isinstance(use_paddle, bool) else use_paddle
        self.easyocr = easyocr.Reader(**init_kwargs_easyocr) if isinstance(use_easy, bool) else use_easy

        # self.paddleocr = use_paddleocr if isinstance(use_paddleocr, paddleocr.PaddleOCR) else paddleocr.PaddleOCR(init_kwargs_paddleocr_defaults)

        # self.paddleocr = _paddleinst or paddleocr.PaddleOCR(**init_kwargs_paddleocr) if use_paddle else None
        # self.easyocr = _easyinst or easyocr.Reader(**init_kwargs_easyocr) if use_easyocr else None

        # if _paddleinst:
        #     self.paddleocr = _paddleinst
        # else:
        #     self.paddleocr = paddleocr.PaddleOCR(**init_kwargs_paddleocr) if use_paddle else None

        self.use_tesseract = use_tesseract

    def __call__(self, image: "Image") -> dict[str, dict[str, list]]:
        image_arr = np.asarray(image)

        prob_results = {}
        text_results = {}

        if self.easyocr != None:
            try:
                easy_result = self._easyocr_ocr(image_arr)
            except:
                breakpoint()

            _, easy_texts, easy_probs = zip(*easy_result) if easy_result else ([], [], [])
            prob_results["easy"], text_results["easy"] = list(easy_probs), list(easy_texts)

        if self.paddleocr != None:
            paddle_result = self._paddleocr_ocr(image_arr)[0]
            paddle_texts, paddle_probs = zip(*[pair[1] for pair in paddle_result]) if paddle_result else ([], [])
            prob_results["paddle"], text_results["paddle"] = list(paddle_probs), list(paddle_texts)

        if self.use_tesseract:
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
