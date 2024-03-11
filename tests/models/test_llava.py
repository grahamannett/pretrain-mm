import unittest

from pretrain_mm.model.llava.llava_processing import convert_bounding_box_from_relative


class TestLlavaPreprocessing(unittest.TestCase):

    def test_bbox_convertions(self):
        test_str = "Provide the bounding box for the element with text: 'Party Size 2 guests'###Assistant: [0.76, 0.58, 0.88, 0.62] Click Action"
        vals = convert_bounding_box_from_relative(test_str, viewport_size=(1080, 1080))
        self.assertEqual(vals, [821, 626, 950, 670])
