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

  const getScoreColor = (score: number) => {
    if (score >= 80) return '#10b981';
    if (score >= 60) return '#f59e0b';
    return '#ef4444';
  };

  const delta = previousScore !== undefined ? score - previousScore : null;

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
          <div className="text-sm text-gray-500">out of 100</div>
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
    </div>
  );
}
