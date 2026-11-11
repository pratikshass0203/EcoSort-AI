import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { classifyImage } from '../../services/api';
import LoadingSpinner from '../../components/common/LoadingSpinner';

const Scanner: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      const url = URL.createObjectURL(f);
      setPreview(url);
    }
  };

  const onSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const result = await classifyImage(file);
      navigate('/result', { state: { result } });
    } catch (e:any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-emerald-500">Scan Waste</h2>
      <input type="file" accept="image/*" onChange={onFileChange} className="file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:bg-emerald-500 file:text-black file:hover:bg-emerald-600" />
      {preview && (
        <div className="relative w-64 h-64">
          <img src={preview} alt="preview" className="object-cover w-full h-full rounded" />
        </div>
      )}
      {error && <div className="text-red-500">{error}</div>}
      <button onClick={onSubmit} disabled={loading || !file} className="px-4 py-2 bg-emerald-500 text-black rounded hover:bg-emerald-600 transition disabled:opacity-50">
        {loading ? 'Classifying...' : 'Classify'}
      </button>
      {loading && <LoadingSpinner />}
    </div>
  );
};

export default Scanner;

