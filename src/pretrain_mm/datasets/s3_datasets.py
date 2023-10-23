from dataclasses import dataclass

import boto3
import botocore
from torch.utils.data import Dataset


@dataclass
class S3Info:
    bucket_name: str


def fetch_image_from_s3(image_url: str):
    pass


class S3Helper:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = boto3.client("s3", config=botocore.client.Config(signature_version=botocore.UNSIGNED))

    def download_file(self, key: str, out_path: str):
        self.client.download_file(self.bucket_name, key, out_path)


class S3Dataset(S3Helper, Dataset):
    pass
