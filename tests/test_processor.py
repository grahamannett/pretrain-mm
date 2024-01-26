import io
import unittest

import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from config.fuyu import FuyuInfo
from pretrain_mm.model.fuyu.processing import FuyuImageProcessor, FuyuProcessor, segment_str
from tests.fixtures.fuyu_fixtures import (
    MODEL_ID,
    default_tokenizer,
    get_kwargs_for_preprocess_with_tokenizer_info,
    get_model_and_patch,
    image,
    input_label,
    input_string,
    input_string_with_label,
)

# no tags
string1 = "Given the following: 10, 20, 30, 40"
# 1 tag
string2 = "Given <0x04><0x00>48, 28, 108, 118<0x01> what is|ENDOFTEXT|"
# bad tag
string3 = "Given <0x04><0x00>48, 108, 118<0x01> What is"
# three_tags
string4 = "input <0x00>12, 34, 56, 78<0x01> <0x02> 90, 12 <0x03> extra box <0x00>100, 200, 150, 12<0x01>"
# three_tags_extra
string5 = "input <0x00>12, 34, 56, 78<0x01> <0x02> 90, 12 <0x03> extra box <0x00>100, 200, 150, 12<0x01> other"
# intermingled tags + start of tag but no end/vals + and no spaces
string6 = "input 1 <0x00> <0x02> 90, 12 <0x03> box then <0x00>12, 34, 56, 78<0x01> extra box <0x00>11,21,15,12<0x01>"


class TestBaseProcessor(unittest.TestCase):
    def test_processor(self):
        processor = FuyuProcessor.from_pretrained(FuyuInfo.model_name)

        image = torch.rand(3, 1280, 1080)
        text = "Task: Find JetBlue career openings in New York Previous Actions Next Action:CLICK  @ <box>172, 4215, 234, 4233</box>"

        data = processor(text=text, images=image)

        self.assertEquals(data.input_ids.ndim, 2)
        self.assertIsInstance(data.image_patches, list)
        self.assertIsInstance(data.image_patches[0], torch.Tensor)
        self.assertEquals(data.image_patches[0].shape[0], 1)


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        self.image = Image.open("tests/fixtures/screenshot0.png")

    def test_prepare(self):
        image_processor = FuyuImageProcessor()
        image, image_info = image_processor.prepare_image(self.image)

        self.assertEqual(image_info, {"height": 1080, "width": 1280, "channels": 3})
        self.assertEqual(image.shape, (1, 3, 1080, 1290))
        self.assertEqual(image.dtype, torch.float32)

    def test_patchify(self):
        image_processor = FuyuImageProcessor()

        patch_size = 4
        test_patch = torch.ones((3, patch_size, patch_size))

        # make image with patch sizes and then stack so that i can assert the patches are in the right place
        image = torch.cat([test_patch, 2 * test_patch, 3 * test_patch], dim=2)
        image = torch.cat([image, 3 + image], dim=1)

        image = image.unsqueeze(0)
        patches = image_processor.patchify_image(image, 4, 4)

        self.assertTrue((patches[0, 0] == 1).all())
        self.assertTrue((patches[0, -1] == 6).all())

        image, original_image_size = image_processor.prepare_image(self.image)

        patchified_image = image_processor.patchify_image(image)
        self.assertEqual(patchified_image.shape, (1, 1548, 2700))
        self.assertEqual(patchified_image.dtype, torch.float32)

    def test_image_tokens(self):
        image_processor = FuyuImageProcessor()

        image_placeholder_id = 71011
        image_newline_id = 71019

        image, original_size = image_processor.prepare_image(self.image)
        patch_cols = image.shape[-1] // image_processor.patch_size
        patch_rows = image.shape[-2] // image_processor.patch_size
        image_patches = image_processor.patchify_image(image, flatten=False)

        self.assertEqual(image_patches.shape[2:], (image_processor.patch_size, image_processor.patch_size, 3))

        image_ids, image_pos_ids = image_processor.make_image_tokens(
            image_placeholder_id=image_placeholder_id,
            image_newline_id=image_newline_id,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            image_patches=image_patches,
        )

        # get original image processor

        original_processor = AutoProcessor.from_pretrained(MODEL_ID)

        original_kwargs = get_kwargs_for_preprocess_with_tokenizer_info(self.image, original_processor)
        original_batch = original_processor.image_processor.preprocess_with_tokenizer_info(**original_kwargs)

        self.assertTrue(torch.equal(image_ids, original_batch.image_input_ids[0][0]))
        self.assertTrue(torch.equal(image_pos_ids, original_batch.image_patch_indices_per_batch[0][0]))
        self.assertTrue(torch.equal(image_pos_ids, original_batch.image_patch_indices_per_subsequence[0][0]))

    def test_preprocess(self):
        image_processor = FuyuImageProcessor()
        image_batch = image_processor.encode_image(self.image)

        image_patches = image_batch.image_patches
        image_ids = image_batch.image_ids
        image_pos_ids = image_batch.image_pos_ids

        self.assertEqual(image_patches.shape[0], len((image_ids != image_processor._image_newline_id).nonzero()))
        self.assertEqual(image_patches.shape[0], len((image_pos_ids != -1).nonzero()))


class TestProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.image = image
        self.image_width, self.image_height = self.image.size[0], self.image.size[1]
        return super().setUp()

    def tearDown(self) -> None:
        self.image.close()
        return super().tearDown()

    def test_segment_str(self):
        test1 = segment_str(string1)
        test2 = segment_str(string2)
        test3 = segment_str(string3)
        test4 = segment_str(string4)
        test5 = segment_str(string5)
        test6 = segment_str(string6)

        self.assertEqual(len(test1), 1)
        self.assertEqual(test1[0][1], None)

        self.assertEqual(len(test4), 6)
        self.assertEqual(test4[-1][1], "box")

        self.assertEqual(len(test6), 6)
        self.assertEqual(test4[-1][1], "box")

    def test_transform(self):
        im_processor = FuyuImageProcessor()
        processor = FuyuProcessor(im_processor, tokenizer=default_tokenizer)

        processor.preprocess_text(string2)

    def test_combine(self):
        processor = FuyuProcessor.from_pretrained(MODEL_ID)
        label = input_string_with_label[
            input_string_with_label.index("<box>") : input_string_with_label.index("</box>") + 6
        ]

        batch = processor(text=input_string_with_label, images=self.image, add_bos_token=True)
        encoded_str = processor.post_process_box_coordinates(batch.input_ids[0, -40:])

        decoded_str = processor.decode(encoded_str)
        self.assertTrue(label in decoded_str)

    def test_combine_with_label(self):
        processor = FuyuProcessor.from_pretrained(MODEL_ID)
        batch = processor(
            text=input_string,
            images=image,
            label=input_label,
            add_bos_token=True,
            add_boa_token=True,
            label_add_eos_token=True,
        )
        input_ids = batch.input_ids
        decoded = processor.decode(input_ids[0])

        self.assertIn(
            '|NEWLINE|<s> Given the following HTML provide the bounding box\\n <button backend_node_id="661"></button>\x04\x002753351600\x01|ENDOFTEXT|',
            decoded,
        )

    def test_box_issues(self):
        # batch input ids with wrong number of tokens between tags
        processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        batch = processor(text=input_string_with_label, images=image, add_bos_token=True)

        input_ids = batch.input_ids[0, -7:-1].tolist()
        # make second token be second token and then 3rd token
        input_ids[2:3] = [input_ids[2], input_ids[2] + 1]

        input_ids[-1:] = [71119, 71118, 70536, 70537, 70054, 70603, 71119]
        post_processed_bbox_tokens = processor.post_process_box_coordinates(input_ids)
        decoded_text = processor.decode(post_processed_bbox_tokens)
        breakpoint()


class TestHFCompare(unittest.TestCase):
    def test_tokens(self):
        # compare tokens from my implementation and hf
        # https://huggingface.co/adept/fuyu-8b/discussions/44
        # using this example as image is below resize threshold
        fifth_text_prompt = (
            "Answer the following VQAv2 question based on the image: What type of foods are in the image?"
        )
        fish_image_url = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/fish_carrots.png"
        )
        fish_image_pil = Image.open(io.BytesIO(requests.get(fish_image_url).content))

        # get both processors
        hf_proc = AutoProcessor.from_pretrained(MODEL_ID)

        hf_proc.max_tokens_to_generate = 0
        processor = FuyuProcessor.from_pretrained(MODEL_ID)

        hf_inputs = hf_proc(text=fifth_text_prompt, images=fish_image_pil)
        inputs = processor(text=fifth_text_prompt, images=fish_image_pil, add_bos_token=True, add_boa_token=True)

        self.assertEqual(hf_inputs.keys(), inputs.keys())
        self.assertEqual(hf_inputs.input_ids.shape, inputs.input_ids.shape)

        # they give list of image patches, we give a tensor
        self.assertTrue((hf_inputs.image_patches[0] == inputs.image_patches).all())
        self.assertTrue((hf_inputs.attention_mask == inputs.attention_mask).all())

        # they add extra -1 on image patches indices because they forget to take it off it seems
        self.assertEqual(hf_inputs.image_patches_indices.shape[:-1], inputs.image_patches_indices.shape[:-1])
        self.assertEqual(
            hf_inputs.image_patches_indices.shape[-1] - hf_proc.max_tokens_to_generate,
            inputs.image_patches_indices.shape[-1],
        )

        model = get_model_and_patch()

        same_gen_kwargs = {"temperature": 0.01, "do_sample": True, "max_new_tokens": 10}

        model_outputs = model.generate(**inputs, **same_gen_kwargs)
        hf_proc_model_outputs = model.generate(**hf_inputs, **same_gen_kwargs)

        decoded_model_outputs = processor.batch_decode(model_outputs[:, -10:], skip_special_tokens=True)
        hf_decoded_model_outputs = hf_proc.batch_decode(hf_proc_model_outputs[:, -10:], skip_special_tokens=True)

        diff_gen_kwargs = {"temperature": 1.0, "do_sample": True, "max_new_tokens": 10}
        self.assertEqual(decoded_model_outputs, hf_decoded_model_outputs)

        # if sample with high temp hopefully different
        model_outputs = model.generate(**inputs, **diff_gen_kwargs)
        hf_proc_model_outputs = model.generate(**hf_inputs, **diff_gen_kwargs)

        decoded_model_outputs = processor.batch_decode(model_outputs[:, -10:], skip_special_tokens=True)
        hf_decoded_model_outputs = hf_proc.batch_decode(hf_proc_model_outputs[:, -10:], skip_special_tokens=True)
        self.assertNotEqual(decoded_model_outputs, hf_decoded_model_outputs)
