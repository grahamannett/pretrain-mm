from dataclasses import dataclass

import boto3
import botocore
from torch.utils.data import Dataset


@dataclass
class S3Info:
    bucket_name: str


class S3Helper:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = boto3.client("s3", config=botocore.client.Config(signature_version=botocore.UNSIGNED))

    def download_file(self, key: str, out_path: str, bucket_name: str = None, **kwargs):
        bucket_name = bucket_name or self.bucket_name
        self.client.download_file(bucket_name, key, out_path)


class S3Dataset(S3Helper, Dataset):
    pass
