import React from 'react';

const SplitView = ({ leftContent, rightContent, loading, sliderValue }) => {
  const getModelLabel = (val) => {
    return `BDH Logic Influence: ${val}%`;
  };

  return (
    <div className="mt-6 space-y-4">
      {/* BDH Model Output */}
      <div>
        <h3 className="font-semibold text-white">{getModelLabel(sliderValue)}</h3>
        <div className="mt-2 p-3 bg-[#0d1117] border border-gray-600 rounded-md min-h-[100px] text-gray-300 whitespace-pre-wrap">
          {loading ? <SkeletonLoader /> : leftContent || 'Output will appear here...'}
        </div>
      </div>
      
      {/* GPT-2 Baseline Output */}
      <div>
        <h3 className="font-semibold text-white">Standard GPT-2 (Baseline)</h3>
        <div className="mt-2 p-3 bg-[#0d1117] border border-gray-600 rounded-md min-h-[100px] text-gray-300 whitespace-pre-wrap">
          {loading ? <SkeletonLoader /> : rightContent || 'Output will appear here...'}
        </div>
      </div>
    </div>
  );
};

const SkeletonLoader = () => (
  <div className="animate-pulse space-y-3">
    <div className="h-4 bg-gray-200 rounded w-3/4"></div>
    <div className="h-4 bg-gray-200 rounded w-full"></div>
    <div className="h-4 bg-gray-200 rounded w-5/6"></div>
    <div className="h-4 bg-gray-200 rounded w-4/6"></div>
  </div>
);

export default SplitView;
