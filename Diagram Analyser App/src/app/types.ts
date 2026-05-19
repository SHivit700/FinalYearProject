export type DiagramType = 'system-design' | 'timeline-roadmap';

export type Severity = 'critical' | 'warning' | 'pass';

export interface MetricThreshold {
  critical: number;
  warning: number;
}

export interface MetricResult {
  name: string;
  score: number;
  severity: Severity;
  description: string;
  recommendation: string;
  flaggedLocations: Array<{ x: number; y: number; width: number; height: number }>;
  isDismissed: boolean;
}

export interface AnalysisResult {
  version: number;
  timestamp: string;
  imagePath: string;
  imageData: string;
  compositeScore: number;
  metrics: MetricResult[];
  criticalCount: number;
  warningCount: number;
  aiNarrative?: string;
}

export interface Session {
  id: string;
  name: string;
  diagramType: DiagramType;
  createdAt: string;
  updatedAt: string;
  versions: AnalysisResult[];
  dismissedMetrics: string[];
  customThresholds: Record<string, MetricThreshold>;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export const METRIC_NAMES = [
  'Label Readability',
  'Label Area',
  'Overlap (Crowding)',
  'Edge Clearance',
  'Font Hierarchy',
  'Container Utilisation',
  'Isolated Boxes',
  'Brevity',
  'Whitespace Distribution',
  'Color Harmony',
  'Label Contrast',
  'Cognitive Chunk Density',
  'Orientation Consistency',
] as const;

export type MetricName = typeof METRIC_NAMES[number];

export const DEFAULT_THRESHOLDS: Record<MetricName, MetricThreshold> = {
  'Label Readability': { critical: 50, warning: 70 },
  'Label Area': { critical: 40, warning: 65 },
  'Overlap (Crowding)': { critical: 30, warning: 60 },
  'Edge Clearance': { critical: 45, warning: 70 },
  'Font Hierarchy': { critical: 50, warning: 75 },
  'Container Utilisation': { critical: 40, warning: 65 },
  'Isolated Boxes': { critical: 35, warning: 60 },
  'Brevity': { critical: 50, warning: 70 },
  'Whitespace Distribution': { critical: 45, warning: 68 },
  'Color Harmony': { critical: 55, warning: 75 },
  'Label Contrast': { critical: 50, warning: 72 },
  'Cognitive Chunk Density': { critical: 42, warning: 65 },
  'Orientation Consistency': { critical: 48, warning: 70 },
};
