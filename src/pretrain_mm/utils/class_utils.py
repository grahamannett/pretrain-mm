# was initially just using this for processor/tokenizer constants/special tokens
# cant decide if this should be on TokenizerConstants or each individual constants class
class SingletonMixin:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMixin, cls).__new__(cls, *args, **kwargs)
        return cls._instances[cls]
