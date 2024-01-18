from itertools import cycle
import torch


class BaseConstrainer:
    def __init__(self, tokenizer: callable, steps: list[list[int]] = [], cycle_steps: bool = False):
        self.tokenizer = tokenizer

        self.n_step = 0
        self.steps = cycle(steps) if cycle_steps else steps

    def __call__(self, logits: torch.Tensor):
        return self.constrain_step(logits)

    def get_step(self) -> list[int]:
        # idk if this makes sense to be cycle
        step = next(self.steps) if isinstance(self.steps, cycle) else self.steps[self.n_step]
        self.n_step += 1
        return step

    def reset(self):
        self.n_step = 0

    def setup_steps(self, steps: list[list[int]]):
        self.steps = steps

    def make_step_idxs(self, vals: list[str] | str):
        if isinstance(vals, str):
            vals = [vals]

        return tokenizer.convert_tokens_to_ids(vals)

    def setup_numbers(self, n_range: tuple[int, int] = (0, 1000)):
        return self.make_step_idxs([str(i) for i in range(*n_range)])


class LogitConstrainer(BaseConstrainer):
    """alter logits of idxs by temperature for a given step

    Args:
        BaseConstrainer (_type_): _description_
    """

    def __init__(self, tokenizer: callable):
        super().__init__(tokenizer)
        self.temp = 0.1

    def constrain_step(self, logits: torch.Tensor, num_samples: int = 1):
        logit_step_keys = self._get_logit_step_keys()
        logits[logit_step_keys] = logits[logit_step_keys] / self.temp

        return torch.multinomial(logits=logits, num_samples=num_samples)


class Constrainer(BaseConstrainer):
    def __init__(self, tokenizer: callable):
        super().__init__(tokenizer)

    def constrain_step(self, logits: torch.Tensor, num_samples: int = 1):
        exclude_mask = torch.ones_like(logits, dtype=torch.bool)
        exclude_mask[self._get_logit_step_keys()] = False

        logits[exclude_mask] = 0

        return torch.multinomial(logits, num_samples=num_samples)


if __name__ == "__main__":
    from transformers import AutoProcessor, AutoTokenizer
    from pretrain_mm.model.fuyu import FuyuConstants

    tokenizer = AutoTokenizer.from_pretrained("adept/fuyu-8b")
    proc = AutoProcessor.from_pretrained("adept/fuyu-8b")

    constrain = Constrainer(tokenizer)

    step0 = constrain.make_step_idxs(FuyuConstants.token_bbox_open_string)
    step1_to_4 = [constrain.make_step_idxs([str(i) for i in range(1000)]) for _ in range(4)]
    step5 = constrain.make_step_idxs(FuyuConstants.token_bbox_close_string)
    steps = [step0, *step1_to_4, step5]

    step0_idxs = constrain.make_step_idxs(FuyuConstants.token_bbox_open_string)
