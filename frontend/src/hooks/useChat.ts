import { useState } from "react";

import type { Message } from "@/types";
import type { Chunk, ParsedFile } from "@/types/document";

import { analyzeFile, sendConversationMessage } from "@/services/api";

interface SendMessageOptions {
  message: string;
  files: File[];
}

interface UseChatOptions {
  conversationId: string;
  messages: Message[];
  isLoading: boolean;
  chunks: Chunk[];
  setMessages: (updater: (prev: Message[]) => Message[]) => void;
  setLoading: (id: string, loading: boolean) => void;
  onAddDocument: (document: ParsedFile) => void;
  onGenerateTitle?: (id: string, firstMessage: string) => void;
}

export function useChat({
  conversationId,
  messages,
  isLoading,
  setMessages,
  setLoading,
  onAddDocument,
  onGenerateTitle,
}: UseChatOptions) {
  const [error, setError] = useState<string | null>(null);

  async function sendMessage(payload: SendMessageOptions) {
    const { message, files } = payload;

    const trimmedMessage = message.trim();

    if (!trimmedMessage && files.length === 0) return;

    const isFirstTurn = messages.length === 0;

    setLoading(conversationId, true);
    setError(null);

    /*
     * Show temporary user message immediately while request processes.
     * Replaced by backend database entry upon success.
     */
    const temporaryUserMessage: Message | null = trimmedMessage
      ? {
          id: crypto.randomUUID(),
          role: "user",
          content: trimmedMessage,
          timestamp: new Date(),
        }
      : null;

    if (temporaryUserMessage) {
      setMessages((prev) => [...prev, temporaryUserMessage]);
    }

    /*
     * Trigger frontend title generation on initial turn.
     */
    if (isFirstTurn && onGenerateTitle && trimmedMessage) {
      onGenerateTitle(conversationId, trimmedMessage);
    }

    try {
      if (files.length > 0) {
        /*
         * Analyze attached files via pipeline with active conversation context.
         */
        const analyses = await Promise.all(
          files.map((file) => analyzeFile(file, conversationId)),
        );

        // Update application document list for UI rendering
        analyses.forEach((analysis, i) =>
          onAddDocument({
            id: analysis.document.id,
            filename: analysis.document.filename ?? files[i].name,
            chunks: analysis.chunks,
            mimeType: analysis.document.mime_type,
            documentType: analysis.document.document_type,
            metadata: analysis.document.metadata,
            createdAt: new Date(analysis.document.created_at),
          }),
        );

        if (trimmedMessage) {
          /*
           * Send user prompt. Backend performs vector retrieval against DB.
           */
          const { userMessage, assistantMessage } =
            await sendConversationMessage({
              conversationId,
              content: trimmedMessage,
            });

          /*
           * Replace temporary user message with real PostgreSQL-backed
           * message and append assistant response.
           */
          setMessages((prev) => [
            ...prev.filter((msg) => msg.id !== temporaryUserMessage?.id),
            userMessage,
            assistantMessage,
          ]);
        } else {
          /*
           * Local document summary response for promptless file uploads.
           */
          const summaryText =
            analyses.length === 1
              ? analyses[0].result
              : analyses
                  .map(
                    (analysis, i) =>
                      `## ${files[i].name}\n\n${analysis.result}`,
                  )
                  .join("\n\n---\n\n");

          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: summaryText,
              timestamp: new Date(),
            },
          ]);
        }
      } else {
        /*
         * Standard message turn. Server handles pgvector retrieval and message storage.
         */
        const { userMessage, assistantMessage } = await sendConversationMessage(
          {
            conversationId,
            content: trimmedMessage,
          },
        );

        /*
         * Replace temporary user message with database-backed
         * message and append assistant response.
         */
        setMessages((prev) => [
          ...prev.filter((msg) => msg.id !== temporaryUserMessage?.id),
          userMessage,
          assistantMessage,
        ]);
      }
    } catch (err) {
      /*
       * Backend request failed, clean up temporary user message entry.
       */
      if (temporaryUserMessage) {
        setMessages((prev) =>
          prev.filter((msg) => msg.id !== temporaryUserMessage.id),
        );
      }

      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(conversationId, false);
    }
  }

  return {
    state: {
      messages,
      isLoading,
      error,
    },
    sendMessage,
    clearError: () => setError(null),
  };
}
