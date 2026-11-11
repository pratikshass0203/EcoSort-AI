// src/components/common/LoadingSpinner.tsx
import React from 'react';
import { Loader2 } from 'lucide-react';

const LoadingSpinner: React.FC = () => (
  <div className="flex items-center justify-center py-4 text-emerald-500">
    <Loader2 className="animate-spin w-8 h-8 mr-2" /> Loading...
  </div>
);

export default LoadingSpinner;

