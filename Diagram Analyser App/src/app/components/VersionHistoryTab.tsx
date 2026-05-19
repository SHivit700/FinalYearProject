import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { AnalysisResult } from '../types';
import { Card } from './ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Badge } from './ui/badge';

interface VersionHistoryTabProps {
  versions: AnalysisResult[];
}

export function VersionHistoryTab({ versions }: VersionHistoryTabProps) {
  const [compareVersion1, setCompareVersion1] = useState<number>(
    versions.length >= 2 ? versions.length - 2 : 0
  );
  const [compareVersion2, setCompareVersion2] = useState<number>(versions.length - 1);

  if (versions.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        No version history yet. Upload a diagram to get started.
      </div>
    );
  }

  const chartData = versions.map((v) => ({
    version: `V${v.version}`,
    score: v.compositeScore,
    critical: v.criticalCount,
    warnings: v.warningCount,
  }));

  const version1 = versions[compareVersion1];
  const version2 = versions[compareVersion2];

  const getMetricDiff = (metricName: string) => {
    const metric1 = version1.metrics.find(m => m.name === metricName);
    const metric2 = version2.metrics.find(m => m.name === metricName);
    if (!metric1 || !metric2) return 0;
    return metric2.score - metric1.score;
  };

  const significantChanges = version1 && version2 ? (() => {
    const changes = [];

    const newCriticals = version2.metrics.filter(
      m => m.severity === 'critical' && !m.isDismissed
    ).filter(
      m2 => !version1.metrics.find(m1 => m1.name === m2.name && m1.severity === 'critical')
    );

    const fixedCriticals = version1.metrics.filter(
      m => m.severity === 'critical' && !m.isDismissed
    ).filter(
      m1 => !version2.metrics.find(m2 => m2.name === m1.name && m2.severity === 'critical')
    );

    if (newCriticals.length > 0) {
      changes.push(`${newCriticals.length} new critical issue${newCriticals.length > 1 ? 's' : ''}`);
    }

    if (fixedCriticals.length > 0) {
      changes.push(`${fixedCriticals.length} critical issue${fixedCriticals.length > 1 ? 's' : ''} resolved`);
    }

    const scoreDelta = version2.compositeScore - version1.compositeScore;
    if (Math.abs(scoreDelta) >= 10) {
      changes.push(`Score ${scoreDelta > 0 ? 'increased' : 'decreased'} by ${Math.abs(scoreDelta)} points`);
    }

    return changes;
  })() : [];

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Score Progression</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="version" />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="score"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ r: 5 }}
              activeDot={{ r: 7 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Version Comparison</h2>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <label className="text-sm font-medium mb-2 block">Version 1</label>
            <Select
              value={compareVersion1.toString()}
              onValueChange={(v) => setCompareVersion1(parseInt(v))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {versions.map((v, idx) => (
                  <SelectItem key={idx} value={idx.toString()}>
                    Version {v.version} (Score: {v.compositeScore})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="text-sm font-medium mb-2 block">Version 2</label>
            <Select
              value={compareVersion2.toString()}
              onValueChange={(v) => setCompareVersion2(parseInt(v))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {versions.map((v, idx) => (
                  <SelectItem key={idx} value={idx.toString()}>
                    Version {v.version} (Score: {v.compositeScore})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="border rounded-lg p-4">
            <h3 className="font-medium mb-2">Version {version1.version}</h3>
            <img
              src={version1.imageData}
              alt={`Version ${version1.version}`}
              className="w-full h-48 object-contain border rounded mb-2"
            />
            <div className="text-sm text-gray-600">
              Score: <span className="font-medium">{version1.compositeScore}/100</span>
            </div>
          </div>

          <div className="border rounded-lg p-4">
            <h3 className="font-medium mb-2">Version {version2.version}</h3>
            <img
              src={version2.imageData}
              alt={`Version ${version2.version}`}
              className="w-full h-48 object-contain border rounded mb-2"
            />
            <div className="text-sm text-gray-600">
              Score: <span className="font-medium">{version2.compositeScore}/100</span>
            </div>
          </div>
        </div>

        {significantChanges.length > 0 && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <h3 className="font-medium mb-2">Significant Changes</h3>
            <ul className="space-y-1">
              {significantChanges.map((change, idx) => (
                <li key={idx} className="text-sm text-gray-700">• {change}</li>
              ))}
            </ul>
          </div>
        )}

        <div>
          <h3 className="font-medium mb-3">Metric Differences</h3>
          <div className="space-y-2">
            {version2.metrics.filter(m => !m.isDismissed).map((metric) => {
              const diff = getMetricDiff(metric.name);
              return (
                <div key={metric.name} className="flex items-center justify-between py-2 border-b">
                  <span className="text-sm">{metric.name}</span>
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={
                        diff > 0 ? 'default' : diff < 0 ? 'destructive' : 'secondary'
                      }
                    >
                      {diff > 0 ? '+' : ''}{diff}
                    </Badge>
                    <span className="text-sm text-gray-500 w-24 text-right">
                      {version1.metrics.find(m => m.name === metric.name)?.score || 0} → {metric.score}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </Card>
    </div>
  );
}
