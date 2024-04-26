import urllib.request


train_url = "https://raw.githubusercontent.com/abacaj/fine-tune-mistral/main/data/train.jsonl"
valid_url = "https://raw.githubusercontent.com/abacaj/fine-tune-mistral/main/data/validation.jsonl"
output_dir = "../tmpdata"

# Download train file
def download_data(outpur_dir: str):
    train_file = output_dir + "/train.jsonl"
    urllib.request.urlretrieve(train_url, train_file)

    # Download validation file
    valid_file = output_dir + "/validation.jsonl"
    urllib.request.urlretrieve(valid_url, valid_file)

    return train_file, valid_file



