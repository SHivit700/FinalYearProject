import type { MetricResult } from '../types';
import { Badge } from './ui/badge';

interface MetricsGridProps {
  metrics: MetricResult[];
}

export function MetricsGrid({ metrics }: MetricsGridProps) {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500';
      case 'warning':
        return 'bg-amber-500';
      case 'pass':
        return 'bg-green-500';
      default:
        return 'bg-gray-500';
    }
  };

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
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {metrics.filter(m => !m.isDismissed).map((metric) => (
        <div
          key={metric.name}
          className="border rounded-lg p-4 hover:shadow-md transition-shadow"
        >
          <div className="flex items-start justify-between mb-3">
            <h3 className="font-medium text-sm">{metric.name}</h3>
            <Badge variant={getSeverityBadgeVariant(metric.severity)}>
              {metric.severity}
            </Badge>
          </div>

          <div className="mb-2">
            <div className="flex justify-between text-xs text-gray-600 mb-1">
              <span>Score</span>
              <span className="font-medium">{metric.score}/100</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all ${getSeverityColor(metric.severity)}`}
                style={{ width: `${metric.score}%` }}
              />
            </div>
          </div>

          {metric.severity !== 'pass' && (
            <p className="text-xs text-gray-600 mt-2">
              {metric.description}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}
