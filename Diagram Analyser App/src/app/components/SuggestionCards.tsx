import { X, RotateCcw } from 'lucide-react';
import type { MetricResult, MetricThreshold } from '../types';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Card } from './ui/card';

interface SuggestionCardsProps {
  metrics: MetricResult[];
  onDismiss: (metricName: string) => void;
  onRestore: (metricName: string) => void;
  onUpdateThreshold?: (metricName: string, threshold: MetricThreshold) => void;
}

export function SuggestionCards({
  metrics,
  onDismiss,
  onRestore,
}: SuggestionCardsProps) {
  const nonPassingMetrics = metrics.filter(m => m.severity !== 'pass');
  const dismissedMetrics = metrics.filter(m => m.isDismissed);

  const getSeverityBadgeVariant = (severity: string): "default" | "secondary" | "destructive" | "outline" => {
    switch (severity) {
      case 'critical':
        return 'destructive';
      case 'warning':
        return 'default';
      default:
        return 'secondary';
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">
          Suggestions ({nonPassingMetrics.filter(m => !m.isDismissed).length})
        </h2>
        {dismissedMetrics.length > 0 && (
          <span className="text-sm text-gray-500">
            {dismissedMetrics.length} dismissed
          </span>
        )}
      </div>

      {nonPassingMetrics.filter(m => !m.isDismissed).map((metric) => (
        <Card key={metric.name} className="p-4">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <Badge variant={getSeverityBadgeVariant(metric.severity)}>
                  {metric.severity}
                </Badge>
                <h3 className="font-medium">{metric.name}</h3>
                <span className="text-sm text-gray-500">({metric.score}/100)</span>
              </div>

              <div className="space-y-2">
                <div>
                  <p className="text-sm font-medium text-gray-700">Issue:</p>
                  <p className="text-sm text-gray-600">{metric.description}</p>
                </div>

                <div>
                  <p className="text-sm font-medium text-gray-700">Recommendation:</p>
                  <p className="text-sm text-gray-600">{metric.recommendation}</p>
                </div>

                {metric.flaggedLocations.length > 0 && (
                  <div>
                    <p className="text-sm font-medium text-gray-700">
                      Flagged locations: {metric.flaggedLocations.length}
                    </p>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {metric.flaggedLocations.slice(0, 3).map((loc, idx) => (
                        <span key={idx} className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                          ({loc.x}%, {loc.y}%)
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => onDismiss(metric.name)}
              className="shrink-0"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        </Card>
      ))}

      {dismissedMetrics.length > 0 && (
        <div className="border-t pt-4 mt-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Dismissed Metrics</h3>
          <div className="space-y-2">
            {dismissedMetrics.map((metric) => (
              <div
                key={metric.name}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <span className="text-sm text-gray-600">{metric.name}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onRestore(metric.name)}
                >
                  <RotateCcw className="w-4 h-4 mr-1" />
                  Restore
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {nonPassingMetrics.filter(m => !m.isDismissed).length === 0 && dismissedMetrics.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          All metrics are passing! Great work! 🎉
        </div>
      )}
    </div>
  );
}
