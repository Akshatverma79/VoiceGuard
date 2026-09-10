/**
 * VoiceVeritas — src/api/analyze.ts
 * API client for POST /api/analyze manual audio upload.
 */

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

export interface AnalyzeSuccessResponse {
  prediction: "real" | "spoof";
  spoof_probability: number;
  risk_level: "low" | "medium" | "high";
  risk_score: number;
  chunks_analyzed: number;
  audio_duration_s: number;
  chunk_scores?: number[];
  recommendation: string;
  processing_time_ms: number;
  // Demo override cache fields
  sha256_hash?: string;
  from_cache?: boolean;
  cache_note?: string;
}

export interface AnalyzeErrorResponse {
  detail: string;
}

/**
 * Upload an audio file to POST /api/analyze for deepfake detection.
 *
 * @param file The audio File to upload (WAV, MP3, M4A)
 * @returns Parsed AnalyzeSuccessResponse
 * @throws Error with a user-friendly message
 */
export async function uploadAndAnalyzeAudio(
  file: File,
  signal?: AbortSignal
): Promise<AnalyzeSuccessResponse> {
  const formData = new FormData();
  formData.append("file", file, file.name);

  let response: Response;
  try {
    response = await fetch(`${BACKEND_URL}/api/analyze`, {
      method: "POST",
      body: formData,
      signal,
    });
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("Analysis was cancelled.");
    }
    // Network failure / server offline
    throw new Error("Unable to connect to the local analysis server.");
  }

  if (!response.ok) {
    let errorDetail = "Voice analysis failed. Please try again.";
    try {
      const errorJson = (await response.json()) as AnalyzeErrorResponse;
      if (errorJson && typeof errorJson.detail === "string") {
        errorDetail = errorJson.detail;
      }
    } catch {
      // Use status text or generic fallback
      if (response.status === 503) {
        errorDetail = "AI model is still loading. Please try again in a moment.";
      }
    }
    throw new Error(errorDetail);
  }

  const data = (await response.json()) as AnalyzeSuccessResponse;
  return data;
}
