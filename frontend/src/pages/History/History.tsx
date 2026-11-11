// src/pages/History/History.tsx
import React, { useEffect, useState } from 'react';
import { getHistory, deleteScan } from '../../services/api';
import type { ScanOut } from '../../types';
import LoadingSpinner from '../../components/common/LoadingSpinner';
import { Trash2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const History: React.FC = () => {
  const [scans, setScans] = useState<ScanOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const fetchHistory = async () => {
    try {
      const data = await getHistory();
      setScans(data);
    } catch (e:any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleDelete = async (id: string) => {
    try {
      await deleteScan(id);
      setScans((prev) => prev.filter((s) => s.id !== id));
    } catch (e:any) {
      setError(e.message);
    }
  };

  const goToResult = (scan: ScanOut) => {
    navigate('/result', { state: { result: scan } });
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="text-red-500">{error}</div>;

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-emerald-500">Scan History</h2>
      {scans.length === 0 ? (
        <p className="text-gray-300">No scans recorded.</p>
      ) : (
        <table className="min-w-full bg-forest-600 rounded">
          <thead className="bg-forest-700">
            <tr>
              <th className="px-4 py-2">Time</th>
              <th className="px-4 py-2">Category</th>
              <th className="px-4 py-2">Confidence</th>
              <th className="px-4 py-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            {scans.map((scan) => (
              <tr key={scan.id} className="border-b border-forest-500">
                <td className="px-4 py-2 text-sm">{new Date(scan.timestamp).toLocaleString()}</td>
                <td className="px-4 py-2 text-sm cursor-pointer" onClick={() => goToResult(scan)}>{scan.category}</td>
                <td className="px-4 py-2 text-sm">{Math.round(scan.confidence * 100)}%</td>
                <td className="px-4 py-2">
                  <button onClick={() => handleDelete(scan.id)} className="text-red-400 hover:text-red-200">
                    <Trash2 className="w-5 h-5" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default History;

