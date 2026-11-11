// src/pages/Analytics/Analytics.tsx
import React, { useEffect, useState } from 'react';
import { getAnalytics } from '../../services/api';
import type { AnalyticsData } from '../../types';
import LoadingSpinner from '../../components/common/LoadingSpinner';
import { PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';

const COLORS = ['#50C878', '#2C5F2D', '#FF8C00', '#FF4500', '#808080'];

const Analytics: React.FC = () => {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetch = async () => {
      try {
        const result = await getAnalytics();
        setData(result);
      } catch (e:any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, []);

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="text-red-500">{error}</div>;

  const pieData = data?.by_category
    ? Object.entries(data.by_category).map(([name, value]) => ({ name, value }))
    : [];

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-emerald-500">Analytics</h2>
      <div className="bg-forest-600 p-4 rounded-lg glassmorphism">
        <p className="text-xl">Total Scans: {data?.total_scans ?? 0}</p>
      </div>
      <div className="bg-forest-600 p-4 rounded-lg glassmorphism">
        <h3 className="text-lg font-semibold mb-2 text-emerald-300">Scans by Category</h3>
        {pieData.length > 0 ? (
          <PieChart width={350} height={300}>
            <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
              {pieData.map((_, i) => (
                <Cell key={`cell-${i}`} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        ) : (
          <p className="text-gray-300">No analytics data.</p>
        )}
      </div>
    </div>
  );
};

export default Analytics;

