import json
import boto3
import uuid
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
from fastapi import HTTPException

from config import config
from models import UserDetails

load_dotenv()


class S3Service:
    def __init__(self):
        self.client = boto3.client(
            's3',
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )

    def extract_s3_key_from_url(self, s3_url: str) -> str:
        """Extract S3 key from S3 URL"""
        if s3_url.startswith("s3://"):
            parts = s3_url[5:].split('/', 1)
            if len(parts) > 1:
                return parts[1]
        raise ValueError(f"Invalid S3 URL format: {s3_url}")

    def fetch_json(self, s3_url: str) -> Dict[str, Any]:
        """Fetch JSON data from S3"""
        try:
            s3_key = self.extract_s3_key_from_url(s3_url)

            response = self.client.get_object(Bucket=config.S3_BUCKET_NAME, Key=s3_key)
            json_data = json.loads(response['Body'].read().decode('utf-8'))

            return json_data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching from S3: {str(e)}")

    def extract_user_uid_from_url(self, s3_url: str) -> str:
        """Extract user UID from the S3 URL path"""
        try:
            # URL format: s3://medora/user_data/25468aba-1eca-4357-b53d-d2f0058d148b/03.03.2025/...
            s3_key = self.extract_s3_key_from_url(s3_url)
            path_parts = s3_key.split('/')

            # Find user_data in the path and get the next part (user_uid)
            if 'user_data' in path_parts:
                user_data_index = path_parts.index('user_data')
                if user_data_index + 1 < len(path_parts):
                    return path_parts[user_data_index + 1]

            raise ValueError("Could not extract user_uid from S3 URL path")
        except Exception as e:
            raise ValueError(f"Error extracting user_uid: {str(e)}")

    def upload_response(self, response_content: str, user_details: UserDetails, cnn_response_url: str, chat_id: str) -> str:
        """Upload LLM response to S3 and return the URL"""
        try:
            # Get current date and time
            current_date = datetime.now()
            date_folder = current_date.strftime("%d.%m.%Y")
            date_filename = current_date.strftime("%d:%m:%Y")
            timestamp = current_date.strftime("%H_%M_%S")

            # Extract user_uid from the CNN response URL
            user_uid = self.extract_user_uid_from_url(cnn_response_url)

            # Create the folder structure: user_data/{user_uid}/{date}/results/
            # Updated filename format to include chatId at the end
            filename = f"user_data/{user_uid}/{date_folder}/results/llm_response_{date_filename}_{timestamp}_{chat_id}.json"

            response_data = {
                "timestamp": current_date.isoformat(),
                "user_details": user_details.dict(),
                "llm_response": response_content,
                "chat_id": chat_id,  # Include chatId in the response data
                "approved": False  # Default to False, will be updated by doctor
            }

            self.client.put_object(
                Bucket=config.S3_BUCKET_NAME,
                Key=filename,
                Body=json.dumps(response_data, indent=2),
                ContentType='application/json'
            )

            return f"s3://{config.S3_BUCKET_NAME}/{filename}"

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error uploading to S3: {str(e)}")
