import React from 'react';

/**
 * SplitView component: Displays two side-by-side panels.
 * Props:
 * - leftContent: React node for left panel content
 * - rightContent: React node for right panel content
 */
const SplitView = ({ leftContent, rightContent }) => {
  return (
    <div className="flex w-full max-w-6xl mx-auto mt-6 gap-4">
      <div className="flex-1 p-4 border rounded shadow bg-white">
        <h2 className="text-lg font-semibold mb-2">BDH Model</h2>
        <div>{leftContent}</div>
      </div>
      <div className="flex-1 p-4 border rounded shadow bg-white">
        <h2 className="text-lg font-semibold mb-2">Standard GPT-2</h2>
        <div>{rightContent}</div>
      </div>
    </div>
  );
};

export default SplitView;
