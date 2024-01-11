class Constrainer:
    def __init__(self, tokenizer: callable):
        self.tokenizer = tokenizer

    def setup_numbers(self, num_range: tuple[int, int] = (0, 1000), additional_numbers: list[int] = []):
        num_tokens = [self.tokenizer.encode(str(v), add_special_tokens=False) for v in range(*num_range)]
        num_tokens += [self.tokenizer.vocab(str(v), add_special_tokens=False) for v in additional_numbers]

    def setup_special_tokens(self):
        pass
