import type { Session, DiagramType, AnalysisResult } from '../types';
import { apiFetch } from './api-client';

export async function getAllSessions(): Promise<Session[]> {
  return apiFetch<Session[]>('/api/sessions');
}

export async function getSession(id: string): Promise<Session | null> {
  return apiFetch<Session>(`/api/sessions/${id}`).catch(() => null);
}

export async function createSession(name: string, diagramType: DiagramType): Promise<Session> {
  return apiFetch<Session>('/api/sessions', {
    method: 'POST',
    body: JSON.stringify({ name, diagramType }),
  });
}

export async function updateSession(session: Session): Promise<void> {
  await apiFetch<Session>(`/api/sessions/${session.id}`, {
    method: 'PUT',
    body: JSON.stringify({
      dismissedMetrics: session.dismissedMetrics,
      customThresholds: session.customThresholds,
    }),
  });
}

export async function deleteSession(id: string): Promise<void> {
  await apiFetch(`/api/sessions/${id}`, { method: 'DELETE' });
}

export function exportSessionAsJson(session: Session): string {
  return JSON.stringify(session, null, 2);
}

export function downloadSessionJson(session: Session): void {
  const json = exportSessionAsJson(session);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${session.name.replace(/\s+/g, '-')}-${session.id}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
