/**
 * VoiceGuard — src/api/health.ts
 * Health check API call to the FastAPI backend.
 */

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

export interface HealthResponse {
  status: "ok" | "error";
}

/**
 * Calls GET /health on the FastAPI backend.
 * Returns true if connected, false otherwise.
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(5000), // 5s timeout
    });
    if (!res.ok) return false;
    const data: HealthResponse = await res.json();
    return data.status === "ok";
  } catch {
    return false;
  }
}
