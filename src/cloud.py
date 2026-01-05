import os
from pathlib import Path

import aioboto3
from botocore.config import Config


class CloudBucket:
    def __init__(self, bucket_name: str, max_concurrent: int = 10):
        """
        Initialize CloudBucket with async S3/R2 client.

        Args:
            bucket_name: Name of the R2 bucket
            max_concurrent: Maximum number of concurrent uploads/downloads
        """
        if not bucket_name:
            raise ValueError("Bucket name is required")
        self.bucket_name = bucket_name
        self.max_concurrent = max_concurrent
        self.endpoint_url = (
            f"https://{os.getenv('CF_R2_ACCOUNTID')}.r2.cloudflarestorage.com"
        )
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.config = Config(signature_version="s3v4")
        self.session = aioboto3.Session()
        if self.aws_access_key_id is None or self.aws_secret_access_key is None:
            raise ValueError("AWS access key ID and secret access key are required")

    async def upload_file(self, file_path: Path, s3_key: str):
        """Upload a single file asynchronously"""
        async with self.session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name="auto",
            config=self.config,
        ) as s3_client:
            try:
                print(f"Uploading {file_path} → {s3_key}")
                await s3_client.upload_file(str(file_path), self.bucket_name, s3_key)
            except Exception as e:
                raise RuntimeError(f"Failed to upload {file_path} to {s3_key}: {e}")
