'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface Dataset {
  job_id: string;
  name: string;
  status: string;
  created_at?: string;
  file_type?: string;
}

interface ExistingDatasetsProps {
  backendUrl: string;
  onSelectDataset?: (datasetId: string) => void;
}

const ExistingDatasets: React.FC<ExistingDatasetsProps> = ({ backendUrl, onSelectDataset }) => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        // In the future, replace with actual API endpoint for listing datasets
        const response = await axios.get(`${backendUrl}/api/status`);
        
        if (response.status === 200) {
          // Filter for completed datasets
          const completedDatasets = response.data.filter(
            (job: any) => job.completed && job.status === 'completed'
          );
          setDatasets(completedDatasets);
        } else {
          throw new Error('Failed to fetch datasets');
        }
      } catch (err: any) {
        console.error('Error fetching datasets:', err);
        setError('Failed to load existing datasets');
      } finally {
        setIsLoading(false);
      }
    };

    fetchDatasets();
  }, [backendUrl]);

  if (isLoading) {
    return <div className="text-center p-4">Loading existing datasets...</div>;
  }

  if (error) {
    return <div className="text-center text-red-500 p-4">{error}</div>;
  }

  if (datasets.length === 0) {
    return (
      <div className="text-center text-gray-500 p-4">
        No existing datasets found. Process a file to create your first dataset.
      </div>
    );
  }

  return (
    <div className="mt-6">
      <h3 className="text-lg font-medium mb-3">Existing Datasets</h3>
      <div className="grid gap-3">
        {datasets.map((dataset) => (
          <div
            key={dataset.job_id}
            className="p-3 border rounded-lg bg-white hover:bg-gray-50 cursor-pointer"
            onClick={() => onSelectDataset && onSelectDataset(dataset.job_id)}
          >
            <div className="flex justify-between items-center">
              <div>
                <p className="font-medium">{dataset.name || `Dataset ${dataset.job_id.substring(0, 8)}`}</p>
                <p className="text-xs text-gray-500">
                  Created: {dataset.created_at 
                    ? new Date(dataset.created_at).toLocaleString() 
                    : 'Unknown date'}
                </p>
              </div>
              <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">
                {dataset.status}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ExistingDatasets;