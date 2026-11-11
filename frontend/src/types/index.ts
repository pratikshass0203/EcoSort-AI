// src/types/index.ts
export interface ScanOut {
  id: string;
  filename: string;
  language: string;
  category: string;
  material: string;
  confidence: number;
  details: Record<string, any>;
  timestamp: string;
}

export interface Category {
  id: number;
  name: string;
}

export interface CategoryList {
  categories: Category[];
}

export interface AnalyticsData {
  total_scans: number;
  by_category: Record<string, number>;
}

export interface AssistantMessage {
  message: string;
  language: string;
}

export interface AssistantResponse {
  response: string;
}

export interface DeviceInfo {
  cpu: string;
  accelerators?: string[];
}

export interface Setting {
  key: string;
  value: string;
}

