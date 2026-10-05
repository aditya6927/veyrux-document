from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator
from enum import Enum
from typing import Any

class DocumentType(str, Enum):
    PDF   = 'pdf'
    DOCX  = 'docx'
    IMAGE = 'image'
    TEXT  = 'text'

class ContentType(str, Enum):
    TEXT  = 'text'
    IMAGE = 'image'
    TABLE = 'table'

class ContentBlock(BaseModel):
    content_type: ContentType
    text_content: str | None = None
    byte_content: bytes | None = None
    mime_type:    str | None = None
    metadata:     dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_content_matches_type(self) -> 'ContentBlock':
        if self.content_type == ContentType.TEXT and not self.text_content:
            raise ValueError('TEXT block must have text_content')
        if self.content_type == ContentType.IMAGE and not self.byte_content:
            raise ValueError('IMAGE block must have byte_content')
        if self.content_type == ContentType.TABLE and not self.text_content:
            raise ValueError('TABLE block must have text_content')
        return self

class Page(BaseModel):
    number: int
    blocks: list[ContentBlock]  # rename data → blocks, more descriptive

class ParsedDocument(BaseModel):
    filename:      str | None = None
    mime_type:     str
    document_type: DocumentType
    pages:         list[Page]
    metadata:      dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    content: str
    embedding: list[float]
    source: str
    page_number: int
    chunk_index: int

class DocumentResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )

    id: uuid.UUID
    conversation_id: uuid.UUID
    filename: str | None
    mime_type: str
    document_type: str

    metadata: dict = Field(
        default_factory=dict,
        validation_alias="metadata_",
        serialization_alias="metadata",
    )

    created_at: datetime
    chunks: list[Chunk] = Field(default_factory=list)