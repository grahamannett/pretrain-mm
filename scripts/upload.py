import os

from huggingface_hub import HfApi

# get this from dev config
folder_path = os.environ.get("MODEL_PATH", "/data/graham/models/pretrain-mm/fuyu/masked_output")


api = HfApi()
api.upload_folder(
    folder_path=folder_path,
    repo_id="besiktas/clippy-webagent",
    repo_type="model",
)
