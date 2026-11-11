import React from 'react';
import { Link } from 'react-router-dom';

const Landing: React.FC = () => (
  <div className="flex flex-col items-center justify-center min-h-screen space-y-6">
    <h1 className="text-5xl font-bold text-emerald-500">EcoSort AI</h1>
    <p className="text-lg text-gray-200 max-w-2xl text-center">
      Offline‑first edge AI platform that classifies waste images and provides disposal guidance.
    </p>
    <div className="space-x-4">
      <Link
        to="/scanner"
        className="px-6 py-2 bg-emerald-500 text-black font-semibold rounded hover:bg-emerald-600 transition"
      >
        Scan Waste
      </Link>
      <Link
        to="/dashboard"
        className="px-6 py-2 bg-forest-500 text-white font-semibold rounded hover:bg-forest-600 transition"
      >
        Dashboard
      </Link>
    </div>
  </div>
);

export default Landing;

