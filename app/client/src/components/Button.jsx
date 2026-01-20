import React from 'react';

const Button = ({ children, onClick, disabled, variant = 'primary', className = '' }) => {
  const baseClasses = 'w-full py-3 font-bold rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2';
  const variants = {
    primary: 'text-white bg-blue-600 hover:bg-blue-500 disabled:bg-blue-700 disabled:cursor-not-allowed focus:ring-blue-500',
    secondary: 'text-white border border-blue-600 bg-transparent hover:bg-blue-600 disabled:border-gray-600 disabled:text-gray-400 disabled:cursor-not-allowed focus:ring-blue-600',
  };
  const classes = `${baseClasses} ${variants[variant]} ${className}`;

  return (
    <button onClick={onClick} disabled={disabled} className={classes}>
      {children}
    </button>
  );
};

export default Button;
