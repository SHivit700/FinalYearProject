import type { AnalysisResult, Session, ChatMessage } from '../types';
import { apiFetch, API_URL } from './api-client';

// The OpenAI key lives in .env on the server — no client-side key needed.
export function initializeAI(_key: string): void { /* no-op */ }
export function isAIInitialized(): boolean { return true; }

export async function generateAnalysisNarrative(
  analysis: AnalysisResult,
  _previousAnalysis?: AnalysisResult,
): Promise<string> {
  // Narrative is generated server-side during analysis and returned in aiNarrative.
  return analysis.aiNarrative ?? '';
}

export async function chatWithAI(
  messages: ChatMessage[],
  session: Session,
  _currentAnalysis?: AnalysisResult,
): Promise<string> {
  const lastMessage = messages[messages.length - 1];
  if (!lastMessage) return '';

  const result = await apiFetch<{ reply: string; action: { type: string; metric: string | null } }>(
    `/api/sessions/${session.id}/chat`,
    {
      method: 'POST',
      body: JSON.stringify({ message: lastMessage.content }),
    },
  );

  return result.reply;
}
