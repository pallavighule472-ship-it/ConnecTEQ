import os
import sys
from unittest.mock import MagicMock

# Set required env vars before HR_backend is imported
os.environ.setdefault("AWS_ACCESS_KEY_ID",     "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_REGION",            "us-east-1")
os.environ.setdefault("OPENAI_API_KEY",        "test")
os.environ.setdefault("S3_BUCKET_NAME",        "test-bucket")
os.environ.setdefault("SQS_QUEUE_URL",         "https://sqs.us-east-1.amazonaws.com/123/test")

# Mock boto3 entirely so module-level client creation doesn't hit AWS
mock_boto3 = MagicMock()
mock_boto3.client.return_value = MagicMock()
sys.modules["boto3"] = mock_boto3
