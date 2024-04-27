import unittest

import torch

from pretrain_mm import logger
from pretrain_mm.model.fuyu import FuyuConstants, FuyuProcessor
from pretrain_mm.utils.generate_utils import generate_helper

# fixtures needed
from tests.fixtures.fuyu_fixtures import MODEL_ID, get_model, image


inputs_example = {
    "text": "Answer the following DocVQA question based on the image. \n What website is this?",
    "images": image,
}


class TestGenerateHelper(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = get_model()
        cls.processor = FuyuProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        return super().setUpClass()

    def setUp(self):
        self.max_new_tokens = 20
        self.stop_tokens = [1, 2, 3]
        self.temperature = 0.8
        self.top_k = 5
        self.indices_placeholder = torch.tensor([[-1]])
        self.mask_placeholder = torch.tensor([[1]])
        self.drop_last_of_input = True

    def test_generate_helper(self):
        model, processor = self.model, self.processor
        drop_last_of_input = False

        output = generate_helper(
            model=model,
            processor=processor,
            inputs=inputs_example,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            indices_placeholder=self.indices_placeholder,
            mask_placeholder=self.mask_placeholder,
            drop_last_of_input=drop_last_of_input,
        )

        self.assertIsInstance(output, torch.Tensor)
        decoded_output = processor.decode(output[0])
        logger.info(f"decoded output: {decoded_output}")

    def test_generate_helper_with_stop_tokens(self):
        model, processor = self.model, self.processor
        additional_tokens = ["black", " The", "The", "image", "the image", "The Image", "a"]

        stop_ids = FuyuConstants.get_stop_ids(
            self.processor,
            additional_tokens=additional_tokens + self.stop_tokens,
        )

        output = generate_helper(
            model=model,
            processor=processor,
            inputs=self.inputs,
            max_new_tokens=self.max_new_tokens,
            stop_ids=stop_ids,
            temperature=self.temperature,
            top_k=self.top_k,
            indices_placeholder=self.indices_placeholder,
            mask_placeholder=self.mask_placeholder,
            drop_last_of_input=self.drop_last_of_input,
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, torch.Size([1, self.max_new_tokens]))


class TestHFGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = get_model()
        cls.processor = FuyuProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        return super().setUpClass()

    def setUp(self) -> None:
        self.max_new_tokens = 50
        return super().setUp()

    def test_generate(self):
        inputs = self.processor(**inputs_example, add_bos_token=True, add_boa_token=True)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded_output = self.processor.decode(output[0])
        breakpoint()

    def test_generate_force_ids(self):
        force_words = [FuyuConstants.bbox_open_string, FuyuConstants.bbox_close_string] + [str(i) for i in range(999)]
        force_words_ids = self.processor.tokenizer(force_words, add_special_tokens=False).input_ids

        inputs = self.processor(**inputs_example, add_bos_token=True, add_boa_token=True)
        output = self.model.sample(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            force_words_ids=force_words_ids,
            num_beams=1,
            # num_return_sequences=1,
        )
        decoded_output = self.processor.decode(output[0, inputs.input_ids.shape[-1]])
        print(decoded_output)


if __name__ == "__main__":
    unittest.main()
