import React from 'react';

/**
 * HemisphereSlider component: A range slider (0-100) with labels "Logic (Left Brain)" and "Creativity (Right Brain)".
 * Styled with a blue to pink gradient.
 * Props:
 * - value: number (slider value)
 * - onChange: function (event handler for slider change)
 */
const HemisphereSlider = ({ value, onChange }) => {
  return (
    <div className="w-full max-w-xl mx-auto px-4">
      <div className="flex justify-between mb-1 text-sm font-medium text-gray-700">
        <span>Logic (Left Brain)</span>
        <span>Creativity (Right Brain)</span>
      </div>
      <input
        type="range"
        min="0"
        max="100"
        value={value}
        onChange={onChange}
        className="w-full h-3 rounded-lg appearance-none cursor-pointer bg-gradient-to-r from-blue-500 to-pink-500"
        style={{
          backgroundSize: '100% 100%',
        }}
      />
    </div>
  );
};

export default HemisphereSlider;
