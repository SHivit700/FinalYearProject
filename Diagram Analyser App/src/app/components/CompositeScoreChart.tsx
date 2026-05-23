import { Cell, Pie, PieChart, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface CompositeScoreChartProps {
  score: number;
  previousScore?: number;
}

export function CompositeScoreChart({ score, previousScore }: CompositeScoreChartProps) {
  const data = [
    { name: 'Score', value: score },
    { name: 'Remaining', value: 100 - score },
  ];

  const getScoreColor = (s: number) => {
    if (s >= 80) return '#10b981'; // green-500
    if (s >= 70) return '#22c55e'; // green-400
    if (s >= 60) return '#f59e0b'; // amber-500
    if (s >= 45) return '#f97316'; // orange-500
    return '#ef4444';              // red-500
  };

  const delta = previousScore !== undefined ? score - previousScore : null;

  const scoreLegend = [
    { range: '80–100', label: 'Excellent',       className: 'text-green-700' },
    { range: '70–79',  label: 'Good',            className: 'text-green-600' },
    { range: '60–69',  label: 'Needs attention', className: 'text-amber-700' },
    { range: '45–59',  label: 'At risk',         className: 'text-orange-700' },
    { range: '0–44',   label: 'Critical',        className: 'text-red-700' },
  ];

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative w-48 h-48">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={0}
              dataKey="value"
              startAngle={90}
              endAngle={-270}
            >
              <Cell fill={getScoreColor(score)} />
              <Cell fill="#e5e7eb" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className="text-4xl font-bold">{score}</div>
        </div>
      </div>

      {delta !== null && (
        <div className={`flex items-center gap-1 text-sm font-medium ${
          delta > 0 ? 'text-green-600' : delta < 0 ? 'text-red-600' : 'text-gray-600'
        }`}>
          {delta > 0 ? (
            <TrendingUp className="w-4 h-4" />
          ) : delta < 0 ? (
            <TrendingDown className="w-4 h-4" />
          ) : (
            <Minus className="w-4 h-4" />
          )}
          <span>
            {delta > 0 ? '+' : ''}{delta} {delta !== 0 ? 'from last version' : 'no change'}
          </span>
        </div>
      )}

      <div className="w-full mt-1">
        <p className="text-xs text-gray-500 font-medium mb-2 text-center">Score guide</p>
        <div className="grid grid-cols-2 gap-1">
          {scoreLegend.map(({ range, label, className }, i) => (
            <div
              key={range}
              className={`flex items-center justify-between gap-1 px-1.5 py-0.5 rounded border bg-white border-gray-100 ${i === 4 ? 'col-span-2' : ''}`}
            >
              <span className={`text-xs font-medium ${className}`}>{label}</span>
              <span className="text-xs text-gray-400">{range}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
