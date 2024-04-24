from .fuyu_constants import FuyuConstants
from .fuyu_model import FuyuForCausalLM, PatchedFuyuForCausalLM
from .fuyu_processing import FuyuProcessor


# If you want to use flash attention in the model, uncomment the following line
# from .persimmon_model import PersimmonAttention

MODEL_ID: str = "adept/fuyu-8b"
