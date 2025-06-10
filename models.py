from pydantic import BaseModel
from typing import Dict, Any, Optional

class UserDetails(BaseModel):
    blood_type: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    bio: Optional[str] = None

# In your models.py file, update the RequestBody class:
class RequestBody(BaseModel):
    user_details: UserDetails
    cnn_response_url: str
    chat_id: str

class WeaviateSearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 5
    alpha: float = 0.5

class ProcessingResponse(BaseModel):
    status: str
    detected_condition: str
    confidence: float
    response_url: str
    message: str

class HealthResponse(BaseModel):
    status: str
    service: str
