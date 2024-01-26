import random
import unittest
from itertools import chain

import torch
import transformers

from pretrain_mm.model.fuyu import FuyuForCausalLM
from pretrain_mm.model.input_merger import make_placeholder_idxs
from pretrain_mm.model.model_utils import ModifiedOutputMixin


# use this to test against the torch version
def _make_placeholder_idxs(input_ids: torch.Tensor, placeholder_id: int, flatten: bool = True):
    """for all the image, go through and create the indexes
    e.g.
    for idxs:     0   1   2   3    4  5  6  7    8   9  10  11  12
    input_ids = [[1, 99, 99, 99,  99, 5, 6, 99, 99, 99, 10, 11,  0]]
    we want [[0, 1, 5], [0, 7, 10]]

    Args:
        input_ids (torch.Tensor): _description_
        token_id (int): _description_
    """
    all_idxs = []
    for batch_idx, input_id in enumerate(input_ids):
        sample_idxs = []

        start, end = None, None
        for t_idx, token_id in enumerate(input_id):
            if (start == None) and (token_id == placeholder_id):
                start = t_idx
            if token_id == placeholder_id:
                end = t_idx + 1
            if (start != None) and (token_id != placeholder_id):
                sample_idxs.append([batch_idx, start, end])
                start, end = None, None
        if start != None:
            sample_idxs.append([batch_idx, start, end])
        all_idxs.append(sample_idxs)

    if flatten:
        all_idxs = list(chain(*all_idxs))

    return all_idxs


import timeit


class TestInputMerger(unittest.TestCase):
    def setUp(self):
        self.placeholder_id = 100

    def test_input_merger(self):
        placeholder_id = self.placeholder_id
        input_ids = torch.randint(0, 99, (4, 100))

        # example where images would be
        input_ids[0, 5:10] = placeholder_id
        input_ids[0, 50:60] = placeholder_id

        input_ids[1, 3:60] = placeholder_id
        input_ids[1, 62:70] = placeholder_id

        # check if it gets end
        input_ids[2, 50:] = placeholder_id

        # check if it gets beginning
        input_ids[3, 0:10] = placeholder_id

        image_idxs = make_placeholder_idxs(input_ids, placeholder_id)

        # i am pretty sure this is correct so test against it
        base_image_idxs = _make_placeholder_idxs(input_ids, placeholder_id)

        self.assertTrue(image_idxs.tolist() == base_image_idxs)

        # should have 6 images
        self.assertTrue(image_idxs.shape[0] == 6)

    def test_input_merger_speed(self):
        placeholder_id = self.placeholder_id
        seq_len = 1000
        batch_size = 4

        def make_s_e(s1, s2):
            return random.randint(*s1), random.randint(*s2)

        def get_input_ids():
            input_ids = torch.randint(0, 99, (batch_size, seq_len))

            # started this func thinking i wanted to test image placement better
            # but its not really necessary
            min_images = 1
            max_images = 5
            s1, s2 = make_s_e((3, 6), (9, 15))
            input_ids[0, s1:s2] = placeholder_id
            input_ids[0, 50:60] = placeholder_id

            input_ids[1, 3:60] = placeholder_id
            input_ids[1, 62:70] = placeholder_id

            # check if it gets end
            input_ids[2, 50:] = placeholder_id

            # check if it gets beginning
            input_ids[3, 0:10] = placeholder_id
            return input_ids

        def test_base():
            inp = get_input_ids()
            _make_placeholder_idxs(inp, placeholder_id)

        def test_torch():
            inp = get_input_ids()
            make_placeholder_idxs(inp, placeholder_id)

        torch_time = timeit.timeit(test_torch, number=100)
        list_time = timeit.timeit(test_base, number=100)
        print("torch time:", torch_time)
        print("base time:", list_time)
        print("torch is {}x faster".format(list_time / torch_time))


class TestExtra(unittest.TestCase):
    def test_increase_model_output(self):
        num_layers = 2

        class ModelCls(FuyuForCausalLM, ModifiedOutputMixin):
            pass

        fuyu_config = transformers.AutoConfig.from_pretrained("adept/fuyu-8b")
        fuyu_config.num_hidden_layers = num_layers
        fuyu_config.text_config.num_hidden_layers = num_layers
        # model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", config=fuyu_config, device_map="auto")
        model = ModelCls(config=fuyu_config)
        model.increase_output_size(1)

        model = model.to("cuda")

        input_tensor = torch.randint(100, 1000, (1, 500), device=model.device)
        input_tensor = input_tensor.to(model.device)

        outputs = model._forward(input_ids=input_tensor, labels=input_tensor, output_hidden_states=True)

        breakpoint()
