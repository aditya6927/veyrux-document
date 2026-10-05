import type { Conversation, Message } from "@/types";
import type { Chunk, Document } from "@/types/document";

const API_BASE = import.meta.env.VITE_API_URL;

/* -------------------------------------------------------------------------- */
/*                                Document API                                */
/* -------------------------------------------------------------------------- */

export interface BackendDocument {
  id: string;
  conversation_id: string;
  filename: string | null;
  mime_type: string;
  document_type: string;
  metadata?: Record<string, unknown>;
  created_at: string;
  chunks?: Chunk[];
}

/**
 * Handles file upload, persistence, and analysis.
 * Passes target conversation ID to tie uploaded file to session.
 */
export async function analyzeFile(
  file: File,
  conversationId: string,
): Promise<{ result: string; chunks: Chunk[]; document: BackendDocument }> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(
    `${API_BASE}/analyze?conversation_id=${encodeURIComponent(conversationId)}`,
    {
      method: "POST",
      body: formData,
    },
  );

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Failed to analyze file");
  }

  const data = await response.json();

  return {
    result: data.result,
    chunks: data.chunks ?? [],
    document: data.document,
  };
}

/* -------------------------------------------------------------------------- */
/*                              Conversation API                              */
/* -------------------------------------------------------------------------- */

interface BackendMessage {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

interface BackendConversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages?: BackendMessage[];
  documents?: BackendDocument[];
}

/**
 * Maps backend message structure to the frontend Message format.
 */
function mapMessage(message: BackendMessage): Message {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    timestamp: new Date(message.created_at),
  };
}

/**
 * Maps backend document structure to frontend Document format.
 */
function mapDocument(doc: BackendDocument): Document {
  return {
    id: doc.id,
    filename: doc.filename ?? "Untitled Document",
    mimeType: doc.mime_type,
    documentType: doc.document_type,
    metadata: doc.metadata ?? {},
    createdAt: new Date(doc.created_at),
    chunks: doc.chunks ?? [],
  };
}

/**
 * Maps backend conversation payload to frontend Conversation model.
 */
function mapConversation(conversation: BackendConversation): Conversation {
  return {
    id: conversation.id,
    title: conversation.title,
    messages: (conversation.messages || []).map(mapMessage),
    documents: (conversation.documents || []).map(mapDocument),
  };
}

/**
 * Fetch all conversations from backend storage.
 */
export async function getConversations(): Promise<Conversation[]> {
  const response = await fetch(`${API_BASE}/conversations/`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Failed to load conversations");
  }

  const data: BackendConversation[] = await response.json();

  return data.map(mapConversation);
}

/**
 * Create a new conversation instance on backend storage.
 */
export async function createConversation(
  title = "New Conversation",
): Promise<Conversation> {
  const response = await fetch(
    `${API_BASE}/conversations/?title=${encodeURIComponent(title)}`,
    {
      method: "POST",
    },
  );

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Failed to create conversation");
  }

  const data: BackendConversation = await response.json();

  return mapConversation(data);
}

/**
 * Fetch details for a specific conversation, including messages and attached documents.
 */
export async function getConversation(
  conversationId: string,
): Promise<Conversation> {
  const response = await fetch(`${API_BASE}/conversations/${conversationId}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Failed to load conversation");
  }

  const data: BackendConversation = await response.json();

  return mapConversation(data);
}

/**
 * Delete a conversation by ID.
 */
export async function deleteConversation(
  conversationId: string,
): Promise<void> {
  const response = await fetch(`${API_BASE}/conversations/${conversationId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Failed to delete conversation");
  }
}

/* -------------------------------------------------------------------------- */
/*                                 Message API                                */
/* -------------------------------------------------------------------------- */

interface SendConversationMessageOptions {
  conversationId: string;
  content: string;
}

interface SendMessageResponsePayload {
  user_message: BackendMessage;
  assistant_message: BackendMessage;
}

/**
 * Sends a message to a specific conversation endpoint.
 * Backend handles history aggregation and response generation.
 */
export async function sendConversationMessage({
  conversationId,
  content,
}: SendConversationMessageOptions): Promise<{
  userMessage: Message;
  assistantMessage: Message;
}> {
  const response = await fetch(
    `${API_BASE}/conversations/${conversationId}/messages`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ content }),
    },
  );

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Failed to send message");
  }

  const data: SendMessageResponsePayload = await response.json();

  return {
    userMessage: mapMessage(data.user_message),
    assistantMessage: mapMessage(data.assistant_message),
  };
}
