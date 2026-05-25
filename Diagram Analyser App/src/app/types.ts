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
  llmAnalysis?: { where: string; howToFix: string };
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

export interface MetricDefinition {
  subtitle: string;
  whatItMeasures: string;
  whyItMatters: string;
}

export const METRIC_DEFINITIONS: Record<MetricName, MetricDefinition> = {
  'Label Readability': {
    subtitle: 'How easy your text labels are to read',
    whatItMeasures: 'Checks whether text labels are large enough, not overlapping, and have enough contrast to be legible at a glance.',
    whyItMatters: "If labels are hard to read, viewers spend more time deciphering words than understanding the diagram's meaning.",
  },
  'Label Area': {
    subtitle: 'Whether labels fit neatly inside their boxes',
    whatItMeasures: "Measures how much of each shape's area is used by its label — too little means wasted space, too much means text is cramped.",
    whyItMatters: 'Labels that overflow or are too small relative to their container look unpolished and can confuse the reader.',
  },
  'Overlap (Crowding)': {
    subtitle: 'How much elements are piling on top of each other',
    whatItMeasures: 'Detects shapes, arrows, or labels that overlap or are placed too close together.',
    whyItMatters: 'Overlapping elements make a diagram look cluttered and cause viewers to misread connections between components.',
  },
  'Edge Clearance': {
    subtitle: "How close content gets to the diagram's edges",
    whatItMeasures: 'Checks whether shapes or labels are positioned too near the outer boundary of the canvas.',
    whyItMatters: 'Elements pressed against the edge give a rushed, unfinished impression and can be cut off when printed or exported.',
  },
  'Font Hierarchy': {
    subtitle: "Whether text sizes clearly show what's important",
    whatItMeasures: 'Looks at the variety and consistency of font sizes used across headings, labels, and annotations.',
    whyItMatters: 'Consistent size differences guide the eye — viewers instantly know which text is a title, which is a label, and which is a note.',
  },
  'Container Utilisation': {
    subtitle: 'How well shapes use their available space',
    whatItMeasures: 'Assesses whether bounding boxes and containers are appropriately sized relative to their contents.',
    whyItMatters: 'Oversized empty containers waste space; undersized ones feel cramped and hard to read.',
  },
  'Isolated Boxes': {
    subtitle: 'Shapes that are floating with no connections',
    whatItMeasures: 'Finds shapes or nodes that have no arrows, lines, or connections linking them to the rest of the diagram.',
    whyItMatters: "An unconnected element looks like a mistake — readers wonder if it belongs there or if something is missing.",
  },
  'Brevity': {
    subtitle: 'Whether labels get to the point quickly',
    whatItMeasures: 'Counts the average word length of labels — long labels slow reading and clutter the diagram.',
    whyItMatters: 'Wordy labels force readers to stop and read sentences rather than glancing at a quick keyword, breaking visual flow.',
  },
  'Whitespace Distribution': {
    subtitle: 'How evenly empty space is spread across the diagram',
    whatItMeasures: 'Measures whether blank space is balanced throughout the canvas, or whether one corner is packed while another is empty.',
    whyItMatters: 'Uneven whitespace creates a visual imbalance — one area feels overcrowded while another looks forgotten.',
  },
  'Color Harmony': {
    subtitle: 'Whether the colours work well together',
    whatItMeasures: 'Evaluates whether the colour palette is consistent and similar categories of elements use matching colours.',
    whyItMatters: 'A jarring or random colour scheme distracts from the content; harmonious colours help group related elements visually.',
  },
  'Label Contrast': {
    subtitle: 'How well text stands out from its background',
    whatItMeasures: "Calculates the contrast ratio between label text colour and the shape's fill colour.",
    whyItMatters: 'Low contrast text is the most common accessibility failure — hard to read for everyone, nearly impossible for users with visual impairments.',
  },
  'Cognitive Chunk Density': {
    subtitle: 'How visually complex and crowded your diagram is',
    whatItMeasures: 'Estimates how many distinct visual "chunks" of information are packed into each area of the canvas.',
    whyItMatters: "Too many elements in one region overloads working memory — viewers can't process it all at once and give up or misunderstand.",
  },
  'Orientation Consistency': {
    subtitle: 'Whether shapes and arrows all point the same way',
    whatItMeasures: 'Checks that shapes, arrows, and flow elements follow a consistent directional pattern (e.g. all left-to-right, or all top-to-bottom).',
    whyItMatters: 'Inconsistent orientation forces viewers to constantly reorient themselves, making the diagram harder to follow as a process.',
  },
};

export function getScoreLabel(score: number): { label: string; className: string } {
  if (score >= 80) return { label: 'Excellent',       className: 'text-green-700 bg-green-50 border-green-200' };
  if (score >= 70) return { label: 'Good',            className: 'text-green-600 bg-green-50 border-green-200' };
  if (score >= 60) return { label: 'Needs attention', className: 'text-amber-700 bg-amber-50 border-amber-200' };
  if (score >= 45) return { label: 'At risk',         className: 'text-orange-700 bg-orange-50 border-orange-200' };
  return               { label: 'Critical',           className: 'text-red-700 bg-red-50 border-red-200' };
}
