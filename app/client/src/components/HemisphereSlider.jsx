import React from 'react';

/**
 * HemisphereSlider component: A range slider (0-100) with labels "Logic (Left Brain)" and "Creativity (Right Brain)".
 * Styled with a blue to pink gradient.
 * Props:
 * - value: number (slider value)
 * - onChange: function (event handler for slider change)
 */
const HemisphereSlider = ({ value, onChange }) => {
  const gradientStyle = {
    // Blue (#2563eb) fills from the left as Logic increases.
    // The background/empty part is Gray (#4b5563).
    background: `linear-gradient(to right, #2563eb ${value}%, #4b5563 ${value}%)`,
  };

  return (
    <div className="w-full">
      <div className="flex justify-between text-sm font-medium text-gray-400 mb-1">
        <span>Creativity (GPT-2)</span>
        <span>{value}% Logic</span>
        <span>Logic (BDH)</span>
      </div>
      <input
        type="range"
        min="0"
        max="100"
        value={value}
        onChange={onChange}
        className="w-full h-2 rounded-lg appearance-none cursor-pointer slider-thumb"
        style={gradientStyle}
        aria-label="Logic to Creativity slider"
      />
    </div>
  );
};

export default HemisphereSlider;
