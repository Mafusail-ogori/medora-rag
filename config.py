import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Config:
    # Weaviate
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "http://127.0.0.1:8000")

    # AWS
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME")

    # LLM
    LLM_MODEL: str = os.getenv("LLM_MODEL", "huggingface/HuggingFaceH4/zephyr-7b-beta")

    # Validation
    @classmethod
    def validate(cls):
        required_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "S3_BUCKET_NAME"
        ]
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


config = Config()
