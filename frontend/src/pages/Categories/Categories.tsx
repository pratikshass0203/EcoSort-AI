// src/pages/Categories/Categories.tsx
import React, { useEffect, useState } from 'react';
import { getCategories } from '../../services/api';
import type { Category } from '../../types';
import LoadingSpinner from '../../components/common/LoadingSpinner';

const Categories: React.FC = () => {
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
    <div>
      <h2 className="text-2xl font-bold text-emerald-500 mb-4">Waste Categories</h2>
      <ul className="space-y-2">
        {categories.map((c) => (
          <li key={c.id} className="p-3 bg-forest-600 rounded glassmorphism">
            {c.name}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Categories;

