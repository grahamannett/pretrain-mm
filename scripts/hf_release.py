import os
from dataclasses import dataclass

from huggingface_hub import HfApi, Repository
from simple_parsing import ArgumentParser


@dataclass
class Config:
    model_path: str


parser = ArgumentParser()
parser.add_arguments(Config, dest="config")
args = parser.parse_args()
model_path = args.config.model_path


def upload_model_to_huggingface(model_path, repo_id, repo_type="model", commit_message="Update model"):
    """
    Uploads the trained model to Hugging Face Hub.

    Args:
    - model_path (str): The path to the model directory to upload.
    - repo_id (str): The repository ID on Hugging Face, e.g., "username/repo_name".
    - repo_type (str): The type of repository, e.g., "model".
    - commit_message (str): The commit message for the update.
    """
    # Ensure the model path exists
    if not os.path.isdir(model_path):
        raise ValueError(f"Model path {model_path} does not exist or is not a directory.")

    # Initialize Hugging Face API
    api = HfApi()

    # Check if the repository exists, if not, create it
    try:
        repo_url = api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        print(f"Repository {repo_id} exists. Updated at {repo_url}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Clone the repository locally
    repo_local_path = os.path.join("hf_repos", repo_id.split("/")[-1])
    repo = Repository(local_dir=repo_local_path, clone_from=repo_url)

    # Copy model files to the cloned repository
    for file_name in os.listdir(model_path):
        file_path = os.path.join(model_path, file_name)
        if os.path.isfile(file_path):
            os.system(f"cp {file_path} {repo_local_path}")

    # Commit and push the changes
    repo.git_add(auto_lfs_track=True)
    repo.git_commit(commit_message)
    repo.git_push()

    # do i need?
    # model.push_to_hub(repo_id, config=config)


if __name__ == "__main__":
    model_path = os.environ.get("MODEL_PATH", "/path/to/your/model")
    repo_id = "yourusername/yourmodelname"  # Change this to your Hugging Face username and model name
    upload_model_to_huggingface(model_path, repo_id)
