import { useState, useRef, useEffect } from 'react';
import { AlertCircle, AlertTriangle, MinusCircle, ZoomIn, X } from 'lucide-react';
import type { AnalysisResult, MetricResult, Severity } from '../types';
import { getScoreLabel } from '../types';
import { CompositeScoreChart } from './CompositeScoreChart';
import { MetricsDragBoard } from './MetricsDragBoard';
import { Card } from './ui/card';

interface AnalysisTabProps {
  analysis: AnalysisResult;
  previousAnalysis?: AnalysisResult;
  onDismissMetric: (metricName: string) => void;
  onRestoreMetric: (metricName: string) => void;
  onUpdateSeverity: (metricName: string, newSeverity: Severity) => void;
}

interface OverlayRect { x: number; y: number; width: number; height: number; }
interface ChunkCentroid { cx: number; cy: number; displayLabel: number; }

const VISUALIZATION_CAPTIONS: Record<string, string | string[] | ((metric: MetricResult) => string | string[])> = {
  'Label Readability':       'Boxes highlight labels with low OCR confidence — text may be too small or blurry.',
  'Label Area':              (m) => m.description.includes('sparse') ? [
    'Label coverage is sparse.',
    'The following changes may help improve your score:',
    'Add labels to unlabelled shapes and connectors.',
    'Use more descriptive text to give each element context.',
    'Increase font size so labels take up more visual weight.',
  ] : [
    'Label coverage is too dense.',
    'The following changes may help improve your score:',
    'Remove redundant or duplicate labels.',
    'Shorten label text to key terms only.',
    'Expand the canvas or zoom out to give elements more room.',
    'Merge related labels where possible.',
  ],
  'Overlap (Crowding)':      'Highlighted region shows where labels are most densely packed or overlapping.',
  'Edge Clearance':          'Highlighted strips show the required clear margin along each edge — elements inside must be moved inward.',
  'Font Hierarchy':          (m) => m.severity === 'pass'
    ? `Font hierarchy looks good (score ${m.score.toFixed(0)}/100) — your labels have clearly distinct size tiers.`
    : `Use distinctly different sizes for titles, section headers, and body labels so readers can instantly gauge importance.`,
  'Container Utilisation':   'Highlighted boxes are containers identified as under-utilised or empty.',
  'Isolated Boxes':          'Boxes highlight shapes with no connector lines detected.',
  'Brevity':                 'Boxes highlight diagram elements with overly verbose labels or too many labels crammed into one box.',
  'Whitespace Distribution': 'Spread elements more evenly — move content from crowded areas into empty regions so no corner of the canvas feels abandoned.',
  'Color Harmony':           'Each dot shows where a detected colour falls on the hue wheel. Clustered dots = harmonious palette; spread-out dots = clashing colours.',
  'Label Contrast':          'Boxes highlight labels where text and background contrast is outside the optimal range.',
  'Cognitive Chunk Density': (m) => {
    const count = m.chunkCentroids?.length ?? 0;
    return count > 0
      ? `${count} perceptual chunk${count !== 1 ? 's' : ''} detected — optimal is 3–5. Numbered badges mark each chunk's centre.`
      : 'Estimates how many distinct visual groups the diagram forms — optimal is 3–5.';
  },
  'Orientation Consistency': 'Highlighted region shows where label orientations are most inconsistent.',
};

function resolveCaption(metric: MetricResult): string | string[] | undefined {
  const entry = VISUALIZATION_CAPTIONS[metric.name];
  if (!entry) return undefined;
  return typeof entry === 'function' ? entry(metric) : entry;
}

function CaptionBlock({ cap, className }: { cap: string | string[]; className: string }) {
  if (Array.isArray(cap)) {
    const [header, intro, ...items] = cap;
    return (
      <div className={className}>
        <p className="font-medium mb-0.5">{header}</p>
        {intro && <p className="italic mb-0.5">{intro}</p>}
        <ul className="space-y-0.5">
          {items.map((s, i) => <li key={i}>• {s}</li>)}
        </ul>
      </div>
    );
  }
  return <p className={className}>{cap}</p>;
}

function severityColors(severity: string): { fill: string; stroke: string } {
  switch (severity) {
    case 'critical': return { fill: 'rgba(239,68,68,0.25)',  stroke: 'rgb(239,68,68)' };
    case 'warning':  return { fill: 'rgba(245,158,11,0.25)', stroke: 'rgb(245,158,11)' };
    default:         return { fill: 'rgba(34,197,94,0.25)',  stroke: 'rgb(34,197,94)' };
  }
}

/**
 * Image with SVG overlay rectangles that are always pixel-perfect.
 *
 * The trick: an <svg> with viewBox matching the image's natural dimensions
 * and preserveAspectRatio="xMidYMid meet" uses exactly the same scaling
 * algorithm as CSS object-contain. The two coordinate spaces align perfectly,
 * so rect coordinates derived from the original image pixels land in the
 * right place — no getBoundingClientRect, no ResizeObserver, no CSS tricks.
 */
function DiagramWithOverlays({
  src,
  alt,
  flaggedLocations,
  chunkCentroids,
  severity,
  maxW,
  maxH,
  imgClass,
  outerClass,
  onClick,
  title,
  children,
}: {
  src: string;
  alt: string;
  flaggedLocations: OverlayRect[];
  chunkCentroids?: ChunkCentroid[];
  severity: string;
  maxW: number;
  maxH: number;
  imgClass?: string;
  outerClass?: string;
  onClick?: () => void;
  title?: string;
  children?: React.ReactNode;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);

  // Capture naturalWidth/Height after load. Also check on mount in case
  // the browser already decoded the base64 image before this render.
  useEffect(() => {
    const img = imgRef.current;
    if (img?.naturalWidth) setNat({ w: img.naturalWidth, h: img.naturalHeight });
  }, []);

  const { fill, stroke } = severityColors(severity);

  return (
    <div
      className={`relative flex items-center justify-center overflow-hidden ${outerClass ?? ''}`}
      onClick={onClick}
      title={title}
    >
      <img
        ref={imgRef}
        src={src}
        alt={alt}
        className={`block ${imgClass ?? ''}`}
        style={{ maxWidth: maxW, maxHeight: maxH }}
        onLoad={(e) => {
          const img = e.currentTarget;
          setNat({ w: img.naturalWidth, h: img.naturalHeight });
        }}
      />

      {nat && (flaggedLocations.length > 0 || (chunkCentroids?.length ?? 0) > 0) && (
        <svg
          viewBox={`0 0 ${nat.w} ${nat.h}`}
          preserveAspectRatio="xMidYMid meet"
          className="absolute inset-0 pointer-events-none"
          style={{ width: '100%', height: '100%' }}
        >
          {flaggedLocations.map((loc, i) => (
            <rect
              key={i}
              x={loc.x / 100 * nat.w}
              y={loc.y / 100 * nat.h}
              width={loc.width  / 100 * nat.w}
              height={loc.height / 100 * nat.h}
              fill={fill}
              stroke={stroke}
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
              rx={3}
            />
          ))}
          {chunkCentroids?.map((c) => {
            const cx = c.cx / 100 * nat.w;
            const cy = c.cy / 100 * nat.h;
            const r  = Math.min(nat.w, nat.h) * 0.028;
            return (
              <g key={c.displayLabel}>
                <circle cx={cx} cy={cy} r={r} fill="white" stroke="#1e293b" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
                <text
                  x={cx} y={cy}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={r * 0.9}
                  fontWeight="bold"
                  fill="#1e293b"
                  fontFamily="system-ui, sans-serif"
                >
                  {c.displayLabel}
                </text>
              </g>
            );
          })}
        </svg>
      )}

      {children}
    </div>
  );
}

function hexToHsl(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l * 100];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h = 0;
  if (max === r)      h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
  else if (max === g) h = ((b - r) / d + 2) / 6;
  else                h = ((r - g) / d + 4) / 6;
  return [h * 360, s * 100, l * 100];
}

function hueSpread(palette: string[]): number {
  if (palette.length < 2) return 0;
  const hues = palette.map(h => hexToHsl(h)[0]).sort((a, b) => a - b);
  let maxGap = 0;
  for (let i = 0; i < hues.length; i++) {
    const gap = (hues[(i + 1) % hues.length] - hues[i] + 360) % 360;
    maxGap = Math.max(maxGap, gap);
  }
  return Math.round(360 - maxGap);
}

function HslColorWheelPanel({ palette, size = 200 }: { palette: string[]; size?: number }) {
  const cx = size / 2, cy = size / 2;
  const outerR = size / 2 - 4;
  const innerR = outerR * 0.52;
  const dotRadius = outerR * 0.72;
  const spread = hueSpread(palette);

  return (
    <div className="flex flex-col items-center gap-2 p-3 bg-white">
      <div className="relative flex-shrink-0" style={{ width: size, height: size }}>
        <div
          className="absolute inset-0 rounded-full"
          style={{ background: 'conic-gradient(red, yellow, lime, cyan, blue, magenta, red)' }}
        />
        <svg
          className="absolute inset-0"
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
        >
          <circle cx={cx} cy={cy} r={innerR} fill="white" />
          {palette.map((hex, i) => {
            const angle = (hexToHsl(hex)[0] - 90) * (Math.PI / 180);
            const x = cx + dotRadius * Math.cos(angle);
            const y = cy + dotRadius * Math.sin(angle);
            return (
              <g key={i}>
                <circle cx={x} cy={y} r={size * 0.038} fill="white" />
                <circle cx={x} cy={y} r={size * 0.030} fill={hex}>
                  <title>{hex}</title>
                </circle>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="text-center leading-snug">
        <p className="text-sm font-medium text-gray-700">
          {palette.length} colour{palette.length !== 1 ? 's' : ''} detected
        </p>
        {palette.length > 1 && (
          <p className="text-xs text-gray-500">
            Spread: {spread}° — aim for &lt;90°
          </p>
        )}
      </div>
    </div>
  );
}

export function AnalysisTab({
  analysis,
  previousAnalysis,
  onDismissMetric,
  onRestoreMetric,
  onUpdateSeverity,
}: AnalysisTabProps) {
  const dismissedCount = analysis.metrics.filter(m => m.isDismissed).length;
  const [isImageOpen, setIsImageOpen] = useState(false);
  const [highlightedMetric, setHighlightedMetric] = useState<MetricResult | null>(null);
  const [modalMetric, setModalMetric] = useState<MetricResult | null>(null);

  const openModal = (metric: MetricResult | null) => {
    setModalMetric(metric);
    setIsImageOpen(true);
  };
  const closeModal = () => {
    setIsImageOpen(false);
    setModalMetric(null);
  };

  const _isCCD = (m: MetricResult | null) => m?.name === 'Cognitive Chunk Density';

  // Floating preview overlays — driven by hover, so they can freely change
  const flaggedLocations: OverlayRect[] = _isCCD(highlightedMetric) ? [] :
    (highlightedMetric?.flaggedLocations?.length ?? 0) > 0
      ? highlightedMetric!.flaggedLocations
      : (highlightedMetric?.llmRegions ?? []);
  const severity     = highlightedMetric?.severity ?? 'pass';
  const showFloating =
    highlightedMetric !== null &&
    (flaggedLocations.length > 0 || (highlightedMetric.paletteColors?.length ?? 0) > 0 || (highlightedMetric.chunkCentroids?.length ?? 0) > 0);

  // Modal overlays — frozen to modalMetric so mouse movement can't clear them
  const modalFlaggedLocations: OverlayRect[] = _isCCD(modalMetric) ? [] :
    (modalMetric?.flaggedLocations?.length ?? 0) > 0
      ? modalMetric!.flaggedLocations
      : (modalMetric?.llmRegions ?? []);
  const modalSeverity = modalMetric?.severity ?? 'pass';

  return (
    <div className="space-y-6">

      {/* Floating preview — fixed bottom-right, always visible while hovering */}
      {showFloating && (
        <div
          className="fixed bottom-6 right-6 z-40 shadow-2xl rounded-lg overflow-hidden border-2 border-blue-400 bg-white cursor-zoom-in group"
          onClick={() => openModal(highlightedMetric)}
          title="Click to enlarge"
        >
          <div className="text-xs font-medium text-blue-700 bg-blue-50 px-2 py-1 border-b border-blue-200 truncate max-w-[220px] flex items-center justify-between gap-2">
            <span>{highlightedMetric!.name}</span>
            <ZoomIn className="w-3 h-3 text-blue-400 shrink-0" />
          </div>
          {highlightedMetric!.name === 'Color Harmony' && highlightedMetric!.paletteColors?.length
            ? <HslColorWheelPanel palette={highlightedMetric!.paletteColors} size={180} />
            : (
              <DiagramWithOverlays
                src={analysis.imageData}
                alt="Flagged location preview"
                flaggedLocations={flaggedLocations}
                chunkCentroids={highlightedMetric?.chunkCentroids}
                severity={severity}
                maxW={224}
                maxH={176}
                outerClass="w-56 h-44 bg-white"
              >
                <div className="absolute inset-0 flex items-center justify-center bg-black/0 group-hover:bg-black/25 transition-colors">
                  <ZoomIn className="w-6 h-6 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow" />
                </div>
              </DiagramWithOverlays>
            )
          }
          {resolveCaption(highlightedMetric!) && (
            <CaptionBlock cap={resolveCaption(highlightedMetric!)!} className="text-xs text-gray-600 px-2 py-1.5 border-t border-blue-100 leading-snug max-w-[224px]" />
          )}
        </div>
      )}

      {/* Full-screen modal */}
      {isImageOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
          onClick={closeModal}
        >
          <button
            className="absolute top-4 right-4 text-white bg-black/40 rounded-full p-1.5 hover:bg-black/60"
            onClick={closeModal}
          >
            <X className="w-5 h-5" />
          </button>
          <div
            className="flex flex-col items-center gap-3 max-w-[90vw]"
            onClick={(e) => e.stopPropagation()}
          >
            {modalMetric?.name === 'Color Harmony' && modalMetric.paletteColors?.length
              ? <HslColorWheelPanel palette={modalMetric.paletteColors} size={320} />
              : (
                <DiagramWithOverlays
                  src={analysis.imageData}
                  alt="Analyzed diagram – full size"
                  flaggedLocations={modalFlaggedLocations}
                  chunkCentroids={modalMetric?.chunkCentroids}
                  severity={modalSeverity}
                  maxW={Math.round(window.innerWidth  * 0.9)}
                  maxH={Math.round(window.innerHeight * 0.82)}
                  imgClass="rounded shadow-2xl"
                />
              )
            }
            {modalMetric && resolveCaption(modalMetric) && (
              <CaptionBlock cap={resolveCaption(modalMetric)!} className="text-sm text-white/90 bg-black/50 rounded-lg px-4 py-2 text-center leading-snug max-w-xl" />
            )}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="p-6 flex flex-col items-center justify-center">
          <CompositeScoreChart
            score={analysis.compositeScore}
            previousScore={previousAnalysis?.compositeScore}
          />
        </Card>

        <Card className="lg:col-span-2 p-6">
          <div className="flex items-start gap-4 mb-6">
            <DiagramWithOverlays
              src={analysis.imageData}
              alt="Analyzed diagram"
              flaggedLocations={[]}
              severity={severity}
              maxW={128}
              maxH={128}
              imgClass="border rounded"
              outerClass="w-32 h-32 flex-shrink-0 group cursor-zoom-in"
              onClick={() => openModal(null)}
              title={highlightedMetric
                ? `Hover preview visible bottom-right — click to zoom`
                : 'Click to view full size'}
            >
              <div className="absolute inset-0 flex items-center justify-center rounded bg-black/0 group-hover:bg-black/30 transition-colors">
                <ZoomIn className="w-7 h-7 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow" />
              </div>
            </DiagramWithOverlays>

            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <h2 className="text-lg font-semibold">Version {analysis.version} Analysis</h2>
                <span className={`text-xs font-medium px-2 py-0.5 rounded border ${getScoreLabel(analysis.compositeScore).className}`}>
                  {getScoreLabel(analysis.compositeScore).label}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 text-red-500" />
                  <div>
                    <div className="text-2xl font-bold">{analysis.criticalCount}</div>
                    <div className="text-xs text-gray-500">Critical</div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-amber-500" />
                  <div>
                    <div className="text-2xl font-bold">{analysis.warningCount}</div>
                    <div className="text-xs text-gray-500">Warnings</div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <MinusCircle className="w-5 h-5 text-gray-400" />
                  <div>
                    <div className="text-2xl font-bold">{dismissedCount}</div>
                    <div className="text-xs text-gray-500">Dismissed</div>
                  </div>
                </div>
              </div>
              {analysis.aiNarrative && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <p className="text-sm text-gray-700 leading-relaxed">
                    {analysis.aiNarrative}
                  </p>
                </div>
              )}
            </div>
          </div>
        </Card>
      </div>

      <div>
        <h2 className="text-xl font-semibold mb-4">Metrics Analysis</h2>
        <MetricsDragBoard
          metrics={analysis.metrics}
          onUpdateSeverity={onUpdateSeverity}
          onDismiss={onDismissMetric}
          onRestore={onRestoreMetric}
          onMetricHighlight={setHighlightedMetric}
          onOpenModal={(metric) => { setHighlightedMetric(metric); openModal(metric); }}
          highlightedMetric={highlightedMetric}
        />
      </div>
    </div>
  );
}
