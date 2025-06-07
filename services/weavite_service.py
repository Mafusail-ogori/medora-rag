import requests
from typing import Dict, Any
from fastapi import HTTPException

from config import config
from models import WeaviateSearchRequest


class WeaviateService:
    def __init__(self):
        self.base_url = config.WEAVIATE_URL

    def search_hybrid(self, query: str, limit: int = 3, alpha: float = 0.5) -> Dict[str, Any]:
        """Search Weaviate vector database using hybrid search"""
        try:
            search_url = f"{self.base_url}/v1/vector-db/search/hybrid"

            payload = WeaviateSearchRequest(
                query=query,
                limit=limit,
                alpha=alpha
            ).dict()

            response = requests.post(search_url, json=payload)
            response.raise_for_status()

            return response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error searching Weaviate: {str(e)}")

    def extract_content(self, weaviate_response: Dict[str, Any]) -> str:
        """Extract content from Weaviate response"""
        weaviate_content = ""
        if "documents" in weaviate_response:
            documents = weaviate_response["documents"]
            if isinstance(documents, list) and len(documents) > 0:
                weaviate_content = documents[0].get("content", "")
            elif isinstance(documents, dict):
                weaviate_content = documents.get("content", "")
        return weaviate_content
