import type { AnalysisResult, MetricResult, MetricThreshold, Severity } from '../types';
import { API_URL } from './api-client';

export async function analyzeImage(
  file: File,
  sessionId: string,
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_URL}/api/sessions/${sessionId}/analyze`, {
    method: 'POST',
    body: formData,
    // Do NOT set Content-Type — browser sets it with the multipart boundary
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail?: string }).detail ?? 'Analysis failed');
  }

  return res.json() as Promise<AnalysisResult>;
}

function calculateSeverity(score: number, threshold: MetricThreshold): Severity {
  if (score < threshold.critical) return 'critical';
  if (score < threshold.warning) return 'warning';
  return 'pass';
}

export function recalculateScores(
  metrics: MetricResult[],
  dismissedMetrics: string[],
  customThresholds: Record<string, MetricThreshold>,
): { metrics: MetricResult[]; compositeScore: number } {
  const updatedMetrics = metrics.map((metric) => {
    const customThreshold = customThresholds[metric.name];
    const severity = customThreshold ? calculateSeverity(metric.score, customThreshold) : metric.severity;

    return {
      ...metric,
      severity,
      isDismissed: dismissedMetrics.includes(metric.name),
    };
  });

  const activeMetrics = updatedMetrics.filter(m => !m.isDismissed);
  const compositeScore =
    activeMetrics.length > 0
      ? Math.round(activeMetrics.reduce((sum, m) => sum + m.score, 0) / activeMetrics.length)
      : 0;

  return { metrics: updatedMetrics, compositeScore };
}
