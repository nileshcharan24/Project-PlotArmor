import React from 'react';

const Textarea = React.forwardRef(({ id, value, onChange, placeholder, rows = 4, disabled = false }, ref) => {
  return (
    <textarea
      id={id}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      rows={rows}
      disabled={disabled}
      ref={ref}
      className="w-full p-3 bg-[#0d1117] border border-gray-600 rounded-md mt-1 resize-none focus:ring-2 focus:ring-blue-500 focus:outline-none text-gray-300"
    />
  );
});

export default Textarea;
