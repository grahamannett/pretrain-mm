import unittest

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from pretrain_mm.model.fuyu.processing import FuyuImageProcessor, FuyuProcessor, segment_str

# , segment_str


def get_kwargs_for_preprocess_with_tokenizer_info(images, processor):
    image_encoding = processor.image_processor.preprocess(images, return_tensors="pt")
    batch_images = image_encoding["images"]
    image_unpadded_heights = image_encoding["image_unpadded_heights"]
    image_unpadded_widths = image_encoding["image_unpadded_widths"]
    scale_factors = image_encoding["image_scale_factors"]

    image_placeholder_id = processor.tokenizer("|SPEAKER|", add_special_tokens=False)["input_ids"][1]
    image_newline_id = processor.tokenizer("|NEWLINE|", add_special_tokens=False)["input_ids"][1]
    tensor_batch_images = torch.stack([img[0] for img in batch_images]).unsqueeze(1)

    return {
        # "scale_factors": [scale_factors],
        "image_present": torch.ones(1, 1, 1),
        "image_unpadded_h": torch.tensor([image_unpadded_heights[0]]),
        "image_unpadded_w": torch.tensor([image_unpadded_widths[0]]),
        "image_placeholder_id": image_placeholder_id,
        "image_newline_id": image_newline_id,
        "image_input": tensor_batch_images[0].unsqueeze(0),
        "variable_sized": True,
    }


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        self.image = Image.open("tests/fixtures/screenshot0.png")

    def test_prepare(self):
        image_processor = FuyuImageProcessor()
        image, original_image_size = image_processor.prepare_image(self.image)

        self.assertEqual(original_image_size, {"height": 1080, "width": 1280, "channels": 3})
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
        patches = image_processor.patchify(image, 4, 4)

        self.assertTrue((patches[0, 0] == 1).all())
        self.assertTrue((patches[0, -1] == 6).all())

        image, original_image_size = image_processor.prepare_image(self.image)

        patchified_image = image_processor.patchify(image)
        self.assertEqual(patchified_image.shape, (1, 1548, 2700))
        self.assertEqual(patchified_image.dtype, torch.float32)

    def test_image_tokens(self):
        image_processor = FuyuImageProcessor()

        image_placeholder_id = 71011
        image_newline_id = 71019

        image, original_size = image_processor.prepare_image(self.image)
        patch_cols = image.shape[-1] // image_processor.patch_size
        patch_rows = image.shape[-2] // image_processor.patch_size
        image_patches = image_processor.patchify(image, flatten=False)

        self.assertEqual(image_patches.shape[2:], (image_processor.patch_size, image_processor.patch_size, 3))

        image_ids, image_pos_ids = image_processor.make_image_tokens(
            image_placeholder_id=image_placeholder_id,
            image_newline_id=image_newline_id,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            image_patches=image_patches,
        )

        # get original image processor

        original_processor = AutoProcessor.from_pretrained("adept/fuyu-8b")

        original_kwargs = get_kwargs_for_preprocess_with_tokenizer_info(self.image, original_processor)
        original_batch = original_processor.image_processor.preprocess_with_tokenizer_info(**original_kwargs)

        self.assertTrue(torch.equal(image_ids, original_batch.image_input_ids[0][0]))
        self.assertTrue(torch.equal(image_pos_ids, original_batch.image_patch_indices_per_batch[0][0]))
        self.assertTrue(torch.equal(image_pos_ids, original_batch.image_patch_indices_per_subsequence[0][0]))

    def test_preprocess(self):
        image_processor = FuyuImageProcessor()
        image_batch = image_processor.preprocess(self.image)

        image_patches = image_batch.image_patches
        image_ids = image_batch.image_ids
        image_pos_ids = image_batch.image_pos_ids

        self.assertEqual(image_patches.shape[0], len((image_ids != image_processor._image_newline_id).nonzero()))
        self.assertEqual(image_patches.shape[0], len((image_pos_ids != -1).nonzero()))


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

tokenizer = AutoTokenizer.from_pretrained("adept/fuyu-8b")
im_processor = FuyuImageProcessor()


class TestProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.open("tests/fixtures/screenshot0.png")
        return super().setUp()

    def tearDown(self) -> None:
        self.image.close()
        return super().tearDown()

    def test_stub(self):
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
        # no_tags_str = "Given the following: 10, 20, 30, 40"
        # tag_str = "Given <0x04><0x00>48, 28, 108, 118<0x01> what is|ENDOFTEXT|"
        # bad_tag_str = "Given <0x04><0x00>48, 108, 118<0x01> What is"

        # im_processor = transformers.models.fuyu.FuyuImageProcessor()
        processor = FuyuProcessor(im_processor, tokenizer=tokenizer)

        processor.preprocess_text(string2)

    def test_processor(self):
        # processor = FuyuProcessor(im_processor, tokenizer=tokenizer)
        processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")

        text = 'Given the following HTML provide the bounding box\\n <button backend_node_id="661"> <span backend_node_id="666"> <text backend_node_id="667">Search</text> </span> </button><0x04><box>54, 1066, 102, 1200</box>|ENDOFTEXT|'

        target_size = self.image.size
        batch = processor(text=text, images=self.image, add_bos_token=True)

        # box_tokens = processor.post_process_box_coordinates(
        #     outputs=batch.input_ids[-40:], target_sizes=torch.tensor([1080, 1280])
        # )

        image_width, image_height = self.image.size[0], self.image.size[1]

        post_processed_bbox_tokens = processor.post_process_box_coordinates(
            batch.input_ids[-40:], target_sizes=torch.tensor([])
        )[0]
