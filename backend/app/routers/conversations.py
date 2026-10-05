import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.message_repository import MessageRepository
from app.schemas.chat import (
    ChatMessage,
    ChatRole,
    ChatSendMessageRequest,
    ConversationResponse,
    SendMessageResponse,
)
from app.services.gemini_service import gemini_service
from app.exceptions import ServiceError

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("/", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    title: str = "New Conversation",
    db: AsyncSession = Depends(get_db)
):
    repo = ConversationRepository(db)
    return await repo.create(title=title)


@router.get("/", response_model=list[ConversationResponse])
async def list_conversations(db: AsyncSession = Depends(get_db)):
    repo = ConversationRepository(db)
    return await repo.list_all()


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    repo = ConversationRepository(db)
    conversation = await repo.get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.post("/{conversation_id}/messages", response_model=SendMessageResponse)
async def send_message(
    conversation_id: uuid.UUID,
    request: ChatSendMessageRequest,
    db: AsyncSession = Depends(get_db)
):
    conv_repo = ConversationRepository(db)
    msg_repo = MessageRepository(db)

    conversation = await conv_repo.get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Save incoming user message
    user_msg = await msg_repo.create(
        conversation_id=conversation_id,
        role=ChatRole.USER.value,
        content=request.content
    )

    # Fetch total history for context
    db_messages = await msg_repo.get_by_conversation(conversation_id)
    chat_history = [
        ChatMessage(role=ChatRole(m.role), content=m.content)
        for m in db_messages
    ]

    document_repo = DocumentRepository(db)

    # Generate query embedding off thread for backend similarity retrieval
    query_embedding = await asyncio.to_thread(
        gemini_service.embed_texts,
        [request.content.strip()],
        "RETRIEVAL_QUERY"
    )

    # Query database chunks using pgvector distance search scoped to conversation
    relevant_chunks = await document_repo.search_similar_chunks(
        conversation_id=conversation_id,
        query_embedding=query_embedding[0],
        top_k=5
    )

    # # Temporary verification log for pgvector retrieval
    # print("\n" + "=" * 50)
    # print("RETRIVIAL VERIFICATION LOG:")
    # print(f"Query: '{request.content.strip()}'")
    # print(f"Retrieved {len(relevant_chunks)} chunks from pgvector:")
    # for idx, chunk in enumerate(relevant_chunks, 1):
    #     print(f"  [{idx}] Source: {chunk.source} | Page: {chunk.page_number} | Preview: {chunk.content[:80]}...")
    # print("=" * 50 + "\n")

    try:
        # Run synchronous Gemini call off the main thread
        response_text = await asyncio.to_thread(
            gemini_service.chat,
            chat_history,
            list(relevant_chunks)
        )
    except ServiceError as e:
        raise HTTPException(status_code=502,
                            detail=f"Model gateway error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Critical internal failure: {str(e)}")

    # Save model response
    assistant_msg = await msg_repo.create(
        conversation_id=conversation_id,
        role=ChatRole.ASSISTANT.value,
        content=response_text
    )

    return SendMessageResponse(
        user_message=user_msg,
        assistant_message=assistant_msg
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    repo = ConversationRepository(db)
    conversation = await repo.get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await repo.delete(conversation)