import { useState } from 'react';
import { AlertCircle, AlertTriangle, XCircle, ZoomIn, X } from 'lucide-react';
import type { AnalysisResult, Severity } from '../types';
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

export function AnalysisTab({
  analysis,
  previousAnalysis,
  onDismissMetric,
  onRestoreMetric,
  onUpdateSeverity,
}: AnalysisTabProps) {
  const dismissedCount = analysis.metrics.filter(m => m.isDismissed).length;
  const [isImageOpen, setIsImageOpen] = useState(false);

  return (
    <div className="space-y-6">
      {isImageOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
          onClick={() => setIsImageOpen(false)}
        >
          <button
            className="absolute top-4 right-4 text-white bg-black/40 rounded-full p-1.5 hover:bg-black/60"
            onClick={() => setIsImageOpen(false)}
          >
            <X className="w-5 h-5" />
          </button>
          <img
            src={analysis.imageData}
            alt="Analyzed diagram – full size"
            className="max-w-[90vw] max-h-[90vh] object-contain rounded shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
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
            <button
              className="relative w-32 h-32 flex-shrink-0 group cursor-zoom-in"
              onClick={() => setIsImageOpen(true)}
              title="Click to view full size"
            >
              <img
                src={analysis.imageData}
                alt="Analyzed diagram"
                className="w-full h-full object-contain border rounded"
              />
              <div className="absolute inset-0 flex items-center justify-center rounded bg-black/0 group-hover:bg-black/30 transition-colors">
                <ZoomIn className="w-7 h-7 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow" />
              </div>
            </button>
            <div className="flex-1">
              <h2 className="text-lg font-semibold mb-2">
                Version {analysis.version} Analysis
              </h2>
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="flex items-center gap-2">
                  <XCircle className="w-5 h-5 text-red-500" />
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
                  <AlertCircle className="w-5 h-5 text-gray-400" />
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
        />
      </div>
    </div>
  );
}
