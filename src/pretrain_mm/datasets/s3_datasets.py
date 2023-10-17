from dataclasses import dataclass

import boto3
import botocore


@dataclass
class S3Info:
    bucket_name: str


def fetch_image_from_s3(image_url: str):
    pass


class S3Dataset:
    def get_client(self):
        s3 = boto3.client("s3", config=botocore.client.Config(signature_version=botocore.UNSIGNED))
        return s3

    def download_file(self, client: boto3.client, key: str, out_path: str):
        client.download_file(self.bucket_name, key, out_path)
