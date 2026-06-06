const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8001';

export { API_URL };

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(init?.headers ?? {}),
  };

  const res = await fetch(`${API_URL}${path}`, { ...init, headers });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail?: string }).detail ?? 'API error');
  }

  return res.json() as Promise<T>;
}
