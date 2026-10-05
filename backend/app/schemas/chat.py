import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import Optional

from app.schemas.document import Chunk, DocumentResponse


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    id: Optional[str] = None
    role: ChatRole
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    chunks: Optional[list[Chunk]] = None
    grounding_chunks: Optional[list[Chunk]] = None


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    created_at: datetime


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = Field(default_factory=list)
    documents: list[DocumentResponse] = Field(default_factory=list)


class ChatSendMessageRequest(BaseModel):
    """
    Request schema for user chat messages.
    Backend owns vector retrieval; client only sends the text prompt.
    """
    content: str


class SendMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_message: MessageResponse
    assistant_message: MessageResponse