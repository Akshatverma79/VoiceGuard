/**
 * VoiceVeritas — src/api/overrides.ts
 * API client for the SQLite demo-override cache.
 * Endpoints: GET/POST/DELETE /api/overrides
 */

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

export interface OverrideEntry {
  sha256_hash: string;
  filename: string;
  spoof_probability: number;
  prediction: "real" | "spoof";
  risk_level: "low" | "medium" | "high";
  risk_score: number;
  recommendation: string;
  note: string;
  created_at: string;
}

export interface CreateOverrideRequest {
  sha256_hash: string;
  filename?: string;
  spoof_probability?: number;
  prediction?: "real" | "spoof";
  risk_level?: "low" | "medium" | "high";
  risk_score?: number;
  recommendation?: string;
  note?: string;
}

/** Fetch all stored overrides (newest first). */
export async function fetchOverrides(): Promise<OverrideEntry[]> {
  const res = await fetch(`${BACKEND_URL}/api/overrides`);
  if (!res.ok) throw new Error("Failed to fetch overrides.");
  return res.json() as Promise<OverrideEntry[]>;
}

/** Save an override for the given SHA-256 hash. */
export async function saveOverride(req: CreateOverrideRequest): Promise<void> {
  const body: CreateOverrideRequest = {
    spoof_probability: 0.97,
    prediction: "spoof",
    risk_level: "high",
    risk_score: 0.97,
    recommendation:
      "🚨 AI-generated voice detected (Demo Override). Do NOT proceed — verify through a trusted channel.",
    note: "",
    ...req,
  };
  const res = await fetch(`${BACKEND_URL}/api/overrides`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error((err as { detail: string }).detail ?? "Failed to save override.");
  }
}

/** Remove an override by its SHA-256 hash. */
export async function deleteOverride(sha256Hash: string): Promise<void> {
  const res = await fetch(
    `${BACKEND_URL}/api/overrides/${encodeURIComponent(sha256Hash)}`,
    { method: "DELETE" }
  );
  if (!res.ok) throw new Error("Failed to delete override.");
}
