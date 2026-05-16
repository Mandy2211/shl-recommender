"""
schemas.py  (app/models/schemas.py)
"""

from pydantic import BaseModel
from typing import List, Optional


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str
    description: Optional[str] = None
    job_levels: Optional[List[str]] = None
    duration: Optional[str] = None
    remote: Optional[str] = None
    adaptive: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool