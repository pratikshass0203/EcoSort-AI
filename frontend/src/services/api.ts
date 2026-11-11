// src/services/api.ts
import axios from 'axios';
import type { ScanOut, CategoryList, AnalyticsData, AssistantMessage, AssistantResponse, DeviceInfo, Setting } from '../types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
});

export const classifyImage = async (file: File, language: string = 'en'): Promise<ScanOut> => {
  const form = new FormData();
  form.append('file', file);
  form.append('language', language);
  const response = await api.post<ScanOut>('/classify', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const getHistory = async (): Promise<ScanOut[]> => {
  const response = await api.get<ScanOut[]>('/history');
  return response.data;
};

export const getAnalytics = async (): Promise<AnalyticsData> => {
  const response = await api.get<AnalyticsData>('/analytics');
  return response.data;
};

export const getCategories = async (): Promise<CategoryList> => {
  const response = await api.get<CategoryList>('/categories');
  return response.data;
};

export const assistantChat = async (msg: AssistantMessage): Promise<AssistantResponse> => {
  const response = await api.post<AssistantResponse>('/assistant/chat', msg);
  return response.data;
};

export const getDeviceInfo = async (): Promise<DeviceInfo> => {
  const response = await api.get<DeviceInfo>('/device');
  return response.data;
};

export const getSettings = async (): Promise<Setting[]> => {
  const response = await api.get<Setting[]>('/settings');
  return response.data;
};

export const deleteScan = async (scanId: string): Promise<void> => {
  await api.delete(`/history/${scanId}`);
};

export const healthCheck = async (): Promise<{ status: string }> => {
  const response = await api.get<{ status: string }>('/health');
  return response.data;
};
