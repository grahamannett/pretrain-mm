from typing import Any, TypeAlias

import easyocr
import numpy as np
import paddleocr
import pandas as pd
import pytesseract


# paddleocr returns like (list[Result]) where Result is [Points, (Text, Probability)] and Points are [int, int]:
# [[[[2.0, 3.0], [90.0, 3.0], [90.0, 18.0], [2.0, 18.0]], ('Search Reddit', 0.9748642444610596)]]
# easyocr returns like (list[Result]) where Result is (Points, Text, Probability) and Points are List[int, int]:
# [[[[2, 3], [90, 3], [90, 18], [2, 18]], 'Search Reddit', 0.9748642444610596]]

PaddleOCRResult: TypeAlias = list[list[list[list[float]] | tuple[str, float]]]
EasyOCRResult: TypeAlias = list[list[list[list[int]] | str | float]]
OCRResult: TypeAlias = PaddleOCRResult | EasyOCRResult

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


def get_box_from_points(points: list[tuple[int | float]]) -> list[int]:
    # since points are corners of the box, just get 0th and 2nd (p1 and p3 in image below)
    #
    # p1(x1,y1)-------p2(x2,y1)
    #       |          |
    #       |          |
    # p4(x1,y2)-------p3(x2,y2)
    # p1, p2 = points[0], points[1]
    return list(map(round, [*points[0], *points[2]]))


# Tesseract output data.frame is DataFrame with columns:
#    ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text']
def get_groupedby(df: pd.DataFrame) -> TesseractGroupByResult:
    return df[df.conf != -1].groupby(["page_num", "block_num", "par_num", "line_num"])


def get_text_tesseract(groupby_result: TesseractGroupByResult):
    try:
        results = groupby_result.text.apply(lambda x: " ".join(list(x))).tolist()
    except TypeError:
        results = groupby_result.text.apply(lambda x: " ".join([str(v) for v in x])).tolist()

    return results


def get_probability_tesseract(groupby_result: TesseractGroupByResult):
    # conf score from tesseract is out of 100, the other two are 0-1
    # mean is for words on same block/page/paragraph/line
    return groupby_result.conf.mean().apply(lambda x: x / 100).tolist()


def make_get_type(fn: callable) -> callable:
    def get_type_from_result(result: OCRResult | None):
        if result is None:
            return []

        return [fn(val) for val in result]

    return get_type_from_result


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

        self.paddleocr = paddleocr.PaddleOCR(**kwargs_paddle) if (use_paddle is True) else use_paddle
        self.easyocr = easyocr.Reader(**kwargs_easy) if (use_easy is True) else use_easy

        self.use_tesseract = use_tesseract

    def __call__(self, image: "Image") -> dict[str, dict[str, list]]:
        image_arr = np.asarray(image)

        prob_results = {}
        text_results = {}

        if self.easyocr is not False:
            easy_result = self._easyocr_ocr(image_arr)
            easy_boxes, easy_texts, easy_probs = zip(*easy_result) if easy_result else ([], [], [])
            prob_results["easy"], text_results["easy"] = list(easy_probs), list(easy_texts)

        if self.paddleocr is not False:
            paddle_result = self._paddleocr_ocr(image_arr)[0]
            paddle_texts, paddle_probs = zip(*[pair[1] for pair in paddle_result]) if paddle_result else ([], [])
            prob_results["paddle"], text_results["paddle"] = list(paddle_probs), list(paddle_texts)

        if self.use_tesseract is not False:
            tess_df = self._tesseract_ocr(image_arr)
            tess_grouped = tess_df[tess_df.conf != -1].groupby(["page_num", "block_num", "par_num", "line_num"])
            prob_results["tesseract"], text_results["tesseract"] = (
                get_probability_tesseract(tess_grouped),
                get_text_tesseract(tess_grouped),
            )

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
    np_input: bool = False

    def __init__(self, init_kwargs: dict = {}, ocr_init_func: callable = None):
        self.init_kwargs = init_kwargs
        self.ocr_init_func = ocr_init_func

        # self.reset()

    def __call__(self, image: "Image", **kwargs) -> Any:
        if self.np_input and not isinstance(image, np.ndarray):
            image = np.asarray(image)

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


class SuryaOCRLabler:
    def __init__(self):
        from surya.model.detection import segformer
        from surya.model.recognition.model import load_model
        from surya.model.recognition.processor import load_processor
        from surya.ocr import run_ocr
        # from surya.schema import OCRResult, TextLine

        self.run_ocr = run_ocr

        self.langs = ["en"]
        self.det_processor, self.det_model = segformer.load_processor(), segformer.load_model()
        self.rec_model, self.rec_processor = load_model(), load_processor()

    def __call__(self, image):
        predictions = self.run_ocr(
            [image], [self.langs], self.det_model, self.det_processor, self.rec_model, self.rec_processor
        )
        return predictions


class PaddleOCRLabeler(OCRLabeler):
    np_input = True

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
        use_angle_cls: bool = True,
        show_log: bool = False,
        use_mp: bool = True,
        **kwargs,
    ):
        self.ocr_inst = paddleocr.PaddleOCR(
            lang=lang, use_gpu=use_gpu, use_angle_cls=use_angle_cls, show_log=show_log, use_mp=use_mp, **kwargs
        )
        self.ocr_func = self.ocr_inst.ocr

        self.return_raw = True

    def __call__(self, image: "Image", **kwargs) -> dict[str, list]:
        result = super().__call__(image, **kwargs)

        result = self.fix_results(result)

        if self.return_raw:
            return result

        # this would go from [[boxes0, (text0, prob0)], [boxes1, (text1, prob1)]] to
        # {"text": [text0, text1], "prob": [prob0, prob1]}
        return {
            "bounding_box": get_box_from_points(result),
            "text": get_text_paddleocr(result),
            "prob": get_probability_paddleocr(result),
        }

    def fix_results(self, results):
        # since
        if (len(results) == 1) and (len(results[0][0]) == 2):
            results = results[0]
        else:
            raise ValueError("Results are not in the expected format")

        for idx, res in enumerate(results):
            # convert the bounding box of 4 points to x1,y1,x2,y2
            bounding_box = get_box_from_points(res[0])
            # unpack the text and confidence
            if len(res) > 2:
                raise ValueError("Results contain too many values, should be (points, (text, conf))")

            results[idx] = [bounding_box, *res[1]]

        return results


if __name__ == "__main__":
    from PIL import Image

    image = Image.open("output/tmp_current_image.png")
    cropped_image = Image.open("output/cropped-image.png")
    small_cropped_image = Image.open("output/ez-testcrop.png")
    # labeler = OCRLabeler()
    pocr = PaddleOCRLabeler(use_gpu=True)

    # tes_result = comparer.tesseract_ocr(small_cropped_image)
    # tes_result = pytesseract.image_to_string(small_cropped_image)
    result = pocr(small_cropped_image)
