import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import type { ScanOut } from '../../types';
import LoadingSpinner from '../../components/common/LoadingSpinner';

const Result: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const result = (location.state as { result?: ScanOut })?.result;

  if (!result) {
    // No result provided – go back to scanner
    navigate('/scanner');
    return null;
  }

  const { category, material, confidence, details } = result;
  const lowConfidence = confidence < 0.6;

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h2 className="text-2xl font-bold text-emerald-500">Classification Result</h2>
      {lowConfidence && (
        <div className="p-4 bg-amber-600 text-black rounded">
          <strong>Low confidence ({Math.round(confidence * 100)}%).</strong> Please review the image manually.
        </div>
      )}
      <div className="flex flex-col items-center space-y-4">
        {details?.image_url && (
          <img src={details.image_url} alt="Waste" className="max-h-64 rounded" />
        )}
        <p className="text-lg">Category: <span className="font-medium">{category}</span></p>
        <p className="text-lg">Material: <span className="font-medium">{material}</span></p>
        <p className="text-lg">Confidence: <span className="font-medium">{Math.round(confidence * 100)}%</span></p>
        {details?.disposal_instructions && (
          <div className="p-3 bg-forest-600 rounded w-full">
            <h3 className="font-semibold mb-2 text-emerald-300">Disposal Instructions</h3>
            <p className="text-sm">{details.disposal_instructions}</p>
          </div>
        )}
      </div>
      <button
        onClick={() => navigate('/scanner')}
        className="px-4 py-2 bg-emerald-500 text-black rounded hover:bg-emerald-600 transition"
      >
        Scan Another
      </button>
    </div>
  );
};

export default Result;

