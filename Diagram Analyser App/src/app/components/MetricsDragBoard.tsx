import { DndProvider, useDrag, useDrop } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { GripVertical, X, RotateCcw, AlertCircle, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import type { MetricResult, Severity, MetricName } from '../types';
import { METRIC_DEFINITIONS } from '../types';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';

interface MetricsDragBoardProps {
  metrics: MetricResult[];
  onUpdateSeverity: (metricName: string, newSeverity: Severity) => void;
  onDismiss: (metricName: string) => void;
  onRestore: (metricName: string) => void;
  onMetricHighlight: (metric: MetricResult | null) => void;
  highlightedMetric: MetricResult | null;
}

interface DraggableMetricProps {
  metric: MetricResult;
  onDismiss: (metricName: string) => void;
  onRestore: (metricName: string) => void;
  onMetricHighlight: (metric: MetricResult | null) => void;
}

interface DropZoneProps {
  severity: Severity;
  metrics: MetricResult[];
  onDrop: (metricName: string, newSeverity: Severity) => void;
  onDismiss: (metricName: string) => void;
  onRestore: (metricName: string) => void;
  onMetricHighlight: (metric: MetricResult | null) => void;
}

const ITEM_TYPE = 'metric';

function toQuadrant(cx: number, cy: number): string {
  const h = cx < 33 ? 'left' : cx < 66 ? 'center' : 'right';
  const v = cy < 33 ? 'top' : cy < 66 ? 'middle' : 'bottom';
  if (h === 'center' && v === 'middle') return 'center';
  if (h === 'center') return `${v} area`;
  if (v === 'middle') return `${h} side`;
  return `${v}-${h}`;
}

function DraggableMetric({ metric, onDismiss, onRestore, onMetricHighlight }: DraggableMetricProps) {
  const [{ isDragging }, drag] = useDrag(() => ({
    type: ITEM_TYPE,
    item: { name: metric.name, currentSeverity: metric.severity },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  }), [metric.name, metric.severity]);

  const getSeverityColor = (severity: Severity) => {
    switch (severity) {
      case 'critical': return 'border-l-red-500';
      case 'warning':  return 'border-l-amber-500';
      case 'pass':     return 'border-l-green-500';
    }
  };

  if (metric.isDismissed) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 opacity-60">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="font-medium text-sm text-gray-500">{metric.name}</span>
              <Badge variant="outline" className="text-xs">Dismissed</Badge>
            </div>
            <p className="text-xs text-gray-500">Score: {metric.score}/100</p>
          </div>
          <Button variant="ghost" size="sm" onClick={() => onRestore(metric.name)} className="shrink-0">
            <RotateCcw className="w-3 h-3" />
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={drag}
      className={`bg-white border-l-4 ${getSeverityColor(metric.severity)} border border-gray-200 rounded-lg p-3 cursor-move hover:shadow-md transition-all ${
        isDragging ? 'opacity-50' : ''
      }`}
      onMouseEnter={() => onMetricHighlight(metric)}
      onMouseLeave={() => onMetricHighlight(null)}
    >
      <div className="flex items-start gap-2">
        <GripVertical className="w-4 h-4 text-gray-400 shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 mb-2">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="font-medium text-sm leading-tight">{metric.name}</span>
              </div>
              {METRIC_DEFINITIONS[metric.name as MetricName] && (
                <p className="text-xs text-gray-500 mt-0.5 leading-snug">
                  {METRIC_DEFINITIONS[metric.name as MetricName].subtitle}
                </p>
              )}
            </div>
            <div className="flex items-center gap-1 shrink-0">
              <span className="text-xs font-mono text-gray-500">{metric.score}/100</span>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0 text-gray-400 hover:text-red-500"
                onClick={() => onDismiss(metric.name)}
                title="Dismiss this metric"
              >
                <X className="w-3 h-3" />
              </Button>
            </div>
          </div>

          <p className="text-xs text-gray-600 mb-2 leading-relaxed">{metric.description}</p>

          {metric.severity !== 'pass' && (() => {
            const uniqueQuadrants = Array.from(
              new Set(
                metric.flaggedLocations.map(loc =>
                  toQuadrant(loc.x + loc.width / 2, loc.y + loc.height / 2)
                )
              )
            ).slice(0, 3);
            return (
              <div className="space-y-1.5">
                {metric.llmAnalysis ? (
                  <>
                    <div className="bg-blue-50 border border-blue-200 rounded p-2">
                      <p className="text-xs font-semibold text-blue-800 mb-0.5">Where to look</p>
                      <p className="text-xs text-blue-700">{metric.llmAnalysis.where}</p>
                    </div>
                    <div className="bg-amber-50 border border-amber-200 rounded p-2">
                      <p className="text-xs font-semibold text-amber-800 mb-0.5">How to fix</p>
                      <p className="text-xs text-amber-700">{metric.llmAnalysis.howToFix}</p>
                    </div>
                  </>
                ) : (
                  <div className="bg-blue-50 border border-blue-200 rounded p-2">
                    <p className="text-xs text-gray-700">
                      <span className="font-medium">Fix:</span> {metric.recommendation}
                    </p>
                  </div>
                )}
                {uniqueQuadrants.length > 0 && (
                  <div className="bg-gray-50 border border-gray-200 rounded p-2">
                    <p className="text-xs font-medium text-gray-500 mb-0.5">Areas to check:</p>
                    <ol className="list-decimal list-inside space-y-0.5">
                      {uniqueQuadrants.map((q, i) => (
                        <li key={i} className="text-xs text-gray-600">{q}</li>
                      ))}
                    </ol>
                  </div>
                )}
              </div>
            );
          })()}

          {METRIC_DEFINITIONS[metric.name as MetricName] && (
            <Popover>
              <PopoverTrigger asChild>
                <button
                  aria-label={`Learn more about ${metric.name}`}
                  className="flex items-center gap-0.5 text-blue-400 hover:text-blue-600 transition-colors mt-2"
                >
                  <Info className="w-3 h-3" />
                  <span className="text-xs">Learn more about this metric</span>
                </button>
              </PopoverTrigger>
              <PopoverContent side="top" className="max-w-xs bg-white text-gray-800 border border-gray-200 shadow-lg p-3 rounded-lg">
                <div className="space-y-2 text-xs">
                  <p><span className="font-semibold">What it measures:</span> {METRIC_DEFINITIONS[metric.name as MetricName].whatItMeasures}</p>
                  <p><span className="font-semibold">Why it matters:</span> {METRIC_DEFINITIONS[metric.name as MetricName].whyItMatters}</p>
                </div>
              </PopoverContent>
            </Popover>
          )}
        </div>
      </div>
    </div>
  );
}

function DropZone({ severity, metrics, onDrop, onDismiss, onRestore, onMetricHighlight }: DropZoneProps) {
  const [{ isOver, canDrop }, drop] = useDrop(() => ({
    accept: ITEM_TYPE,
    drop: (item: { name: string; currentSeverity: Severity }) => {
      if (item.currentSeverity !== severity) {
        onDrop(item.name, severity);
      }
    },
    collect: (monitor) => ({
      isOver: monitor.isOver(),
      canDrop: monitor.canDrop(),
    }),
  }), [onDrop, severity]);

  const getSeverityConfig = (severity: Severity) => {
    switch (severity) {
      case 'critical':
        return {
          title: 'Critical Issues',
          description: 'Metrics that are failing and most likely harming readability right now.',
          emptyMessage: 'No critical issues — your diagram is in good shape here.',
          icon: AlertCircle,
          color: 'text-red-600',
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200',
        };
      case 'warning':
        return {
          title: 'Warnings',
          description: 'Metrics below ideal — worth fixing soon to improve clarity.',
          emptyMessage: 'Nothing to flag — this section is clear.',
          icon: AlertTriangle,
          color: 'text-amber-600',
          bgColor: 'bg-amber-50',
          borderColor: 'border-amber-200',
        };
      case 'pass':
        return {
          title: 'Passing',
          description: 'Metrics performing well — no action needed here.',
          emptyMessage: 'No passing metrics yet.',
          icon: CheckCircle,
          color: 'text-green-600',
          bgColor: 'bg-green-50',
          borderColor: 'border-green-200',
        };
    }
  };

  const config = getSeverityConfig(severity);
  const Icon = config.icon;

  return (
    <div className="flex-1 min-w-[300px]">
      <div className={`${config.bgColor} border ${config.borderColor} rounded-lg p-3 mb-3`}>
        <div className="flex items-center gap-2">
          <Icon className={`w-5 h-5 ${config.color}`} />
          <h3 className={`font-semibold ${config.color}`}>{config.title}</h3>
          <Badge variant="secondary" className="ml-auto">{metrics.length}</Badge>
        </div>
        <p className="text-xs text-gray-500 mt-1">{config.description}</p>
      </div>

      <div
        ref={drop}
        className={`min-h-[400px] rounded-lg border-2 border-dashed p-3 space-y-3 transition-colors ${
          isOver && canDrop ? 'border-blue-400 bg-blue-50' : 'border-gray-300 bg-gray-50'
        }`}
      >
        {isOver && canDrop && (
          <div className="flex items-center justify-center h-32 text-blue-600 text-sm font-medium">
            Drop here to reclassify
          </div>
        )}
        {metrics.map((metric) => (
          <DraggableMetric
            key={metric.name}
            metric={metric}
            onDismiss={onDismiss}
            onRestore={onRestore}
            onMetricHighlight={onMetricHighlight}
          />
        ))}
        {metrics.filter(m => !m.isDismissed).length === 0 && !isOver && (
          <div className="flex items-center justify-center h-full min-h-[120px]">
            <p className="text-xs text-gray-400 text-center px-4">{config.emptyMessage}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export function MetricsDragBoard({
  metrics,
  onUpdateSeverity,
  onDismiss,
  onRestore,
  onMetricHighlight,
}: MetricsDragBoardProps) {
  const criticalMetrics = metrics.filter(m => m.severity === 'critical');
  const warningMetrics  = metrics.filter(m => m.severity === 'warning');
  const passMetrics     = metrics.filter(m => m.severity === 'pass');

  return (
    <DndProvider backend={HTML5Backend}>
      <div className="space-y-4">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <p className="text-sm text-gray-700">
            <span className="font-medium">Drag and drop</span> metrics between columns to reclassify their severity.
          </p>
        </div>

        <div className="flex gap-4 overflow-x-auto pb-4">
          <DropZone severity="critical" metrics={criticalMetrics} onDrop={onUpdateSeverity} onDismiss={onDismiss} onRestore={onRestore} onMetricHighlight={onMetricHighlight} />
          <DropZone severity="warning"  metrics={warningMetrics}  onDrop={onUpdateSeverity} onDismiss={onDismiss} onRestore={onRestore} onMetricHighlight={onMetricHighlight} />
          <DropZone severity="pass"     metrics={passMetrics}     onDrop={onUpdateSeverity} onDismiss={onDismiss} onRestore={onRestore} onMetricHighlight={onMetricHighlight} />
        </div>
      </div>
    </DndProvider>
  );
}
