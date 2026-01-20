import React from 'react';

const Card = ({ title, children }) => {
  return (
    <div className="bg-[#161b22] p-6 rounded-lg border border-gray-700 shadow-md">
      {title && <h2 className="text-2xl font-semibold text-white mb-4">{title}</h2>}
      {children}
    </div>
  );
};

export default Card;
