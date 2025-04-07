'use client';

import * as React from 'react';

function getProgressFillStyle(value: number, max: number = 100) {
  const percentage = Math.min(Math.max(value, 0), max) / max * 100;
  return { width: `${percentage}%` };
}

export interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number;
  max?: number;
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value = 0, max = 100, ...props }, ref) => {
    return (
      <div
        ref={ref}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={max}
        aria-valuenow={value}
        className={`w-full bg-gray-200 rounded-full h-4 dark:bg-gray-700 ${className || ''}`}
        {...props}
      >
        <div 
          className="bg-blue-600 h-4 rounded-full transition-all duration-300 ease-in-out" 
          style={getProgressFillStyle(value, max)}
        ></div>
      </div>
    );
  }
);

Progress.displayName = 'Progress';

export { Progress };