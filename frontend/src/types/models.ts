// Type definitions for model structures

interface ModelInfo {
  id: string;
  name?: string;
}

interface ProviderInfo {
  provider: string;
  name: string;
  models: ModelInfo[];
  env_key?: string;
  list_endpoint?: string;
}

export interface AvailableModels {
  [providerKey: string]: ProviderInfo;
}

export interface ProcessingConfig {
  provider: string;
  model: string;
  systemPrompt?: string;
  keywords?: string[];
  language?: string;
  temperature?: number;
  maxTokens?: number;
  addReasoning?: boolean;
  outputFormat?: string;
  processingType?: string;
}