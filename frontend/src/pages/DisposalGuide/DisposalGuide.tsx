// src/pages/DisposalGuide/DisposalGuide.tsx
import React from 'react';
import { useEffect, useState } from 'react';
import LoadingSpinner from '../../components/common/LoadingSpinner';
import { getCategories } from '../../services/api';
import type { Category } from '../../types';

const guideMap: Record<string, string> = {
  Biodegradable: 'Compost or dispose in organic waste bin. Keep dry and avoid contamination.',
  Recyclable: 'Rinse and place in recycling bin. Separate plastics, metal, paper as per local rules.',
  "E-waste": 'Take to an e‑waste collection point. Do not discard in normal trash.',
  Hazardous: 'Handle with gloves, store in labelled container, and bring to hazardous waste facility.',
  Unknown: 'Consult local authority for proper disposal.',
};

const DisposalGuide: React.FC = () => {
  const [categories, setCategories] = useState<Category[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetch = async () => {
      try {
        const data = await getCategories();
        setCategories(data.categories);
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

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-emerald-500">Disposal Guide</h2>
      <ul className="space-y-4">
        {categories.map((c) => (
          <li key={c.id} className="p-4 bg-forest-600 rounded glassmorphism">
            <h3 className="text-xl font-semibold text-emerald-300 mb-2">{c.name}</h3>
            <p>{guideMap[c.name] || 'No specific guidance available.'}</p>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default DisposalGuide;

