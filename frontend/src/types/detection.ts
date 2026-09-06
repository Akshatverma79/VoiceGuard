/**
 * VoiceVeritas — src/types/detection.ts
 * Shared TypeScript types for WebSocket messages.
 */

export type RiskLevel = "low" | "medium" | "high" | "unknown";

/** Sent by the server after a successful deepfake detection inference */
export interface DetectionMessage {
  type: "detection";
  spoof_probability: number;
  prediction: "real" | "spoof";
  risk_score: number;
  risk_level: RiskLevel;
  chunks_analyzed: number;
  history: number[];
  timestamp: string;
}

/** Server status updates (connected, processing, waiting_for_speech, etc.) */
export interface StatusMessage {
  type: "status";
  status:
    | "connected"
    | "listening"
    | "waiting_for_speech"
    | "processing"
    | "timeout";
  native_sample_rate?: number;
  speech_ratio?: number;
}

/** Server error message */
export interface ErrorMessage {
  type: "error";
  message: string;
}

/** Development performance metrics (VERBOSE_PERF=True on backend) */
export interface PerfMessage {
  type: "perf";
  buffer_s: number;
  preprocess_ms: number;
  inference_ms: number;
  total_ms: number;
  chunks: number;
}

export type ServerMessage =
  | DetectionMessage
  | StatusMessage
  | ErrorMessage
  | PerfMessage;

/** Client → Server config message (first message after connect) */
export interface ConfigMessage {
  type: "config";
  sample_rate: number;
}
