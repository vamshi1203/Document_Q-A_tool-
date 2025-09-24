const API_URL = 'http://localhost:8000';

export interface ApiResponse<T> {
  data?: T;
  error?: string;
}

export interface UploadResponse {
  message: string;
  files: string[];
}

export interface AskResponse {
  question: string;
  answer: string;
  sources: string[];
}

export const uploadFiles = async (files: File[]): Promise<ApiResponse<UploadResponse>> => {
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });

  try {
    const response = await fetch(`${API_URL}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      return { error: error.detail || 'Failed to upload files' };
    }

    return { data: await response.json() };
  } catch (error) {
    return { error: 'Network error. Please check your connection.' };
  }
};

export const askQuestion = async (question: string): Promise<ApiResponse<AskResponse>> => {
  try {
    const response = await fetch(`${API_URL}/api/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        question: question
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      return { error: error.detail || 'Failed to get answer' };
    }

    return { data: await response.json() };
  } catch (error) {
    return { error: 'Network error. Please check your connection.' };
  }
};

export const checkHealth = async (): Promise<ApiResponse<{ status: string; message: string }>> => {
  try {
    const response = await fetch(`${API_URL}/api/health`);
    if (!response.ok) {
      return { error: 'API is not available' };
    }
    return { data: await response.json() };
  } catch (error) {
    return { error: 'Cannot connect to the API' };
  }
};
