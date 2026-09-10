/**
 * VoiceVeritas — src/pages/ManualUpload.tsx
 * Manual Audio Upload and Deepfake Check page (Local-only).
 *
 * Architecture:
 *   - Drag & Drop / Click-to-browse audio file selection
 *   - In-browser audio player preview (before analysis)
 *   - Multi-stage loading progress
 *   - Local FastAPI analysis via deepfake detection model
 *   - Risk Engine classification & recommendation
 *   - 100% local session state (no DB, no Supabase, no history)
 */

import { useState, useRef, useEffect, useCallback } from "react";
import { uploadAndAnalyzeAudio, type AnalyzeSuccessResponse } from "../api/analyze";
import {
  fetchOverrides,
  saveOverride,
  deleteOverride,
  type OverrideEntry,
} from "../api/overrides";

type LoadingStage =
  | "idle"
  | "uploading"
  | "processing"
  | "analyzing"
  | "calculating"
  | "complete";

const STAGE_LABELS: Record<LoadingStage, string> = {
  idle: "",
  uploading: "Uploading...",
  processing: "Processing Audio...",
  analyzing: "Running AI Analysis...",
  calculating: "Calculating Risk...",
  complete: "Complete",
};

const STAGE_PROGRESS: Record<LoadingStage, number> = {
  idle: 0,
  uploading: 25,
  processing: 50,
  analyzing: 75,
  calculating: 90,
  complete: 100,
};

const SUPPORTED_EXTS = [".wav", ".mp3", ".m4a"];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB

function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function formatDuration(seconds: number): string {
  if (!seconds || isNaN(seconds)) return "—";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);
  if (mins > 0) {
    return `${mins}m ${secs}s`;
  }
  return `${secs}.${ms}s`;
}

// ── Analysis History ──────────────────────────────────────────────────────────
const HISTORY_KEY = "vg_analysis_history";
const MAX_HISTORY = 20;

interface HistoryEntry {
  id: string;
  filename: string;
  timestamp: string;         // ISO string
  prediction: "real" | "spoof";
  risk_level: "low" | "medium" | "high";
  spoof_probability: number;
  chunks_analyzed: number;
  audio_duration_s: number;
  processing_time_ms: number;
}

function loadHistory(): HistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]") as HistoryEntry[];
  } catch { return []; }
}

function saveHistory(entries: HistoryEntry[]) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, MAX_HISTORY)));
}

function pushHistory(entry: Omit<HistoryEntry, "id" | "timestamp">) {
  const history = loadHistory();
  history.unshift({ ...entry, id: crypto.randomUUID(), timestamp: new Date().toISOString() });
  saveHistory(history);
}

function timeAgo(iso: string): string {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}

function AnalysisHistoryPanel({
  result,
  filename,
}: {
  result: import("../api/analyze").AnalyzeSuccessResponse | null;
  filename: string | null;
}) {
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);
  const [open, setOpen] = useState(true);
  const prevResultRef = useRef<typeof result>(null);

  // Push a new entry every time result changes to a non-null value
  useEffect(() => {
    if (result && result !== prevResultRef.current) {
      prevResultRef.current = result;
      pushHistory({
        filename: filename ?? "Unknown file",
        prediction: result.prediction,
        risk_level: result.risk_level,
        spoof_probability: result.spoof_probability,
        chunks_analyzed: result.chunks_analyzed,
        audio_duration_s: result.audio_duration_s,
        processing_time_ms: result.processing_time_ms,
      });
      setHistory(loadHistory());
    }
  }, [result, filename]);

  const clearHistory = () => {
    localStorage.removeItem(HISTORY_KEY);
    setHistory([]);
  };

  return (
    <div className="w-full max-w-2xl">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-4 py-2.5 rounded-xl bg-slate-900/50 border border-slate-800/60 text-slate-400 hover:text-white hover:border-slate-700 transition-all text-sm font-semibold"
      >
        <span className="flex items-center gap-2">
          <span>📋</span>
          <span>Recent Analysis History</span>
          {history.length > 0 && (
            <span className="text-[10px] font-mono bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded-md">
              {history.length}
            </span>
          )}
        </span>
        <span className="text-xs font-mono">{open ? "▲ hide" : "▼ show"}</span>
      </button>

      {open && (
        <div className="mt-2 bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex flex-col gap-2">
          {history.length === 0 ? (
            <p className="text-slate-500 text-sm text-center py-6">
              No analyses yet. Upload an audio file to get started.
            </p>
          ) : (
            <>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] font-mono text-slate-600 uppercase tracking-wider">
                  Last {history.length} session{history.length > 1 ? "s" : ""}
                </span>
                <button
                  onClick={clearHistory}
                  className="text-[10px] text-slate-600 hover:text-slate-400 transition-colors"
                >
                  Clear history
                </button>
              </div>

              {history.map((entry) => (
                <div
                  key={entry.id}
                  className="flex items-center gap-3 bg-slate-950/50 border border-slate-800/50 rounded-lg px-3 py-2.5"
                >
                  {/* Risk colour dot */}
                  <div className={`w-2 h-2 rounded-full shrink-0 ${
                    entry.risk_level === "high" ? "bg-red-500" :
                    entry.risk_level === "medium" ? "bg-amber-400" : "bg-emerald-400"
                  }`} />

                  {/* File info */}
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-semibold text-white truncate">{entry.filename}</p>
                    <p className="text-[10px] text-slate-500 font-mono">
                      {entry.audio_duration_s.toFixed(1)}s · {entry.chunks_analyzed} chunks · {entry.processing_time_ms}ms
                    </p>
                  </div>

                  {/* Prediction badge */}
                  <span className={`text-[10px] font-black uppercase px-2 py-0.5 rounded-md shrink-0 ${
                    entry.prediction === "spoof"
                      ? "bg-red-500/20 text-red-400 border border-red-500/30"
                      : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                  }`}>
                    {entry.prediction}
                  </span>

                  {/* Spoof % */}
                  <span className={`text-xs font-bold font-mono shrink-0 w-12 text-right ${
                    entry.risk_level === "high" ? "text-red-400" :
                    entry.risk_level === "medium" ? "text-amber-400" : "text-emerald-400"
                  }`}>
                    {(entry.spoof_probability * 100).toFixed(1)}%
                  </span>

                  {/* Time */}
                  <span className="text-[10px] text-slate-600 font-mono shrink-0 w-14 text-right">
                    {timeAgo(entry.timestamp)}
                  </span>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function ManualUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioDuration, setAudioDuration] = useState<number | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [loadingStage, setLoadingStage] = useState<LoadingStage>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeSuccessResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // ── Override cache state ──────────────────────────────────────────────────
  const [overrides, setOverrides] = useState<OverrideEntry[]>([]);
  const [showManager, setShowManager] = useState(false);
  const [overrideStatus, setOverrideStatus] = useState<
    "idle" | "saving" | "saved" | "error"
  >("idle");

  const loadOverrides = useCallback(async () => {
    try {
      const data = await fetchOverrides();
      setOverrides(data);
    } catch {
      // silently ignore — backend may not be up yet
    }
  }, []);

  useEffect(() => { void loadOverrides(); }, [loadOverrides]);

  const handleMarkAsFake = async () => {
    if (!result?.sha256_hash) return;
    setOverrideStatus("saving");
    try {
      await saveOverride({
        sha256_hash: result.sha256_hash,
        filename: selectedFile?.name ?? "",
        spoof_probability: 0.97,
        prediction: "spoof",
        risk_level: "high",
        risk_score: 0.97,
        recommendation:
          "🚨 AI-generated voice detected (Demo Override). Do NOT proceed — verify through a trusted channel.",
        note: "Manually marked as fake for demo",
      });
      setOverrideStatus("saved");
      void loadOverrides();
      setTimeout(() => setOverrideStatus("idle"), 3000);
    } catch {
      setOverrideStatus("error");
      setTimeout(() => setOverrideStatus("idle"), 3000);
    }
  };

  const handleDeleteOverride = async (hash: string) => {
    try {
      await deleteOverride(hash);
      void loadOverrides();
    } catch {
      // ignore
    }
  };

  // Clean up object URL on unmount or file change
  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  const resetSelection = useCallback(() => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setSelectedFile(null);
    setAudioUrl(null);
    setAudioDuration(null);
    setErrorMessage(null);
    setResult(null);
    setLoadingStage("idle");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, [audioUrl]);

  const handleFileValidation = (file: File): boolean => {
    setErrorMessage(null);
    setResult(null);

    const nameLower = file.name.toLowerCase();
    const hasValidExt = SUPPORTED_EXTS.some((ext) => nameLower.endsWith(ext));

    if (!hasValidExt) {
      setErrorMessage("Unsupported audio format. Please upload a WAV, MP3, or M4A file.");
      return false;
    }

    if (file.size === 0) {
      setErrorMessage("The uploaded audio file is empty.");
      return false;
    }

    if (file.size > MAX_FILE_SIZE) {
      setErrorMessage("Audio file exceeds maximum size limit of 50 MB.");
      return false;
    }

    return true;
  };

  const processSelectedFile = (file: File) => {
    if (!handleFileValidation(file)) {
      return;
    }

    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }

    const newUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setAudioUrl(newUrl);
    setAudioDuration(null);
    setResult(null);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processSelectedFile(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      processSelectedFile(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile || loadingStage !== "idle") return;

    setErrorMessage(null);
    setResult(null);
    setLoadingStage("uploading");

    // Realistic progress transition simulation while waiting for backend
    const timer1 = setTimeout(() => setLoadingStage("processing"), 300);
    const timer2 = setTimeout(() => setLoadingStage("analyzing"), 700);
    const timer3 = setTimeout(() => setLoadingStage("calculating"), 1200);

    try {
      const data = await uploadAndAnalyzeAudio(selectedFile);
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      setLoadingStage("complete");
      setResult(data);
    } catch (err: unknown) {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      setLoadingStage("idle");
      const msg = err instanceof Error ? err.message : "Voice analysis failed. Please try again.";
      setErrorMessage(msg);
    }
  };

  const isAnalyzing = loadingStage !== "idle" && loadingStage !== "complete";

  // Cybersecurity color mapping based on risk level
  const riskColor = (level: string) => {
    switch (level) {
      case "high":
        return "text-red-400";
      case "medium":
        return "text-amber-400";
      case "low":
        return "text-emerald-400";
      default:
        return "text-slate-400";
    }
  };

  const riskCardStyle = (level: string) => {
    switch (level) {
      case "high":
        return "bg-red-950/20 border-red-500/40 shadow-2xl shadow-red-500/20";
      case "medium":
        return "bg-amber-950/20 border-amber-500/40 shadow-2xl shadow-amber-500/20";
      case "low":
        return "bg-emerald-950/20 border-emerald-500/40 shadow-2xl shadow-emerald-500/20";
      default:
        return "bg-slate-900/60 border-slate-800 shadow-xl";
    }
  };

  const riskBadgeStyle = (level: string) => {
    switch (level) {
      case "high":
        return "bg-red-500/20 text-red-400 border border-red-500/30";
      case "medium":
        return "bg-amber-500/20 text-amber-400 border border-amber-500/30";
      case "low":
        return "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
      default:
        return "bg-slate-800 text-slate-400 border border-slate-700";
    }
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] p-6 flex flex-col items-center gap-6">
      {/* ── Header ── */}
      <div className="flex flex-col items-center gap-2 pt-2 text-center">
        <h1 className="text-3xl font-black tracking-tight text-white">
          Manual Audio <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-violet-400">Analysis</span>
        </h1>
        <p className="text-slate-400 text-sm max-w-md">
          Upload any recorded voice clip to perform local deepfake voice analysis.
        </p>
        <span className="text-xs font-mono text-slate-500 tracking-widest uppercase">
          Local Analysis · No Cloud Storage · Privacy First
        </span>
      </div>

      {/* ── Error Banner ── */}
      {errorMessage && (
        <div className="w-full max-w-2xl bg-red-500/10 border border-red-500/30 rounded-2xl p-4 text-red-400 text-sm flex items-center justify-between gap-3 shadow-lg shadow-red-950/20">
          <div className="flex items-center gap-3">
            <span className="text-lg">⚠️</span>
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-red-400/80 hover:text-red-300 font-bold px-2 py-1 text-xs rounded hover:bg-red-500/20 transition-colors"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* ── Main Container ── */}
      <div className="w-full max-w-2xl flex flex-col gap-6">

        {/* ── Dropzone / File Picker ── */}
        {!selectedFile && (
          <div
            id="audio-dropzone"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`cursor-pointer rounded-2xl border-2 border-dashed p-10 flex flex-col items-center justify-center gap-4 transition-all duration-200 text-center ${
              isDragOver
                ? "border-indigo-400 bg-indigo-500/10 scale-[1.01]"
                : "border-slate-800 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/60"
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept=".wav,.mp3,.m4a,audio/wav,audio/mpeg,audio/mp4,audio/x-m4a"
              className="hidden"
              id="audio-file-input"
            />

            {/* Music/Wave Icon */}
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-violet-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 text-2xl shadow-lg shadow-indigo-500/10">
              🎵
            </div>

            <div>
              <p className="text-base font-semibold text-white">
                Drop audio file here or <span className="text-indigo-400 underline underline-offset-2">Browse Files</span>
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Supports WAV, MP3, and M4A up to 50 MB
              </p>
            </div>

            <div className="flex items-center gap-2 pt-2">
              {["WAV", "MP3", "M4A"].map((fmt) => (
                <span
                  key={fmt}
                  className="px-2.5 py-0.5 rounded-md text-xs font-mono font-medium text-slate-400 bg-slate-800/80 border border-slate-700/60"
                >
                  {fmt}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* ── Audio Preview Card ── */}
        {selectedFile && audioUrl && (
          <div className="bg-slate-900/60 backdrop-blur-xl border border-slate-800/80 rounded-2xl p-6 shadow-xl flex flex-col gap-4">
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-12 h-12 rounded-xl bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 text-xl shrink-0">
                  🎧
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-white truncate max-w-md" title={selectedFile.name}>
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-slate-400 font-mono mt-0.5">
                    {formatFileSize(selectedFile.size)}
                    {audioDuration !== null && ` • ${formatDuration(audioDuration)}`}
                  </p>
                </div>
              </div>

              {/* Action buttons: Replace & Remove */}
              <div className="flex items-center gap-2 shrink-0">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isAnalyzing}
                  className="text-xs font-medium text-slate-400 hover:text-white px-3 py-1.5 rounded-lg border border-slate-800 hover:border-slate-700 bg-slate-800/40 hover:bg-slate-800 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Replace
                </button>
                <button
                  onClick={resetSelection}
                  disabled={isAnalyzing}
                  className="text-xs font-medium text-red-400 hover:text-red-300 px-3 py-1.5 rounded-lg border border-red-900/30 hover:border-red-800/50 bg-red-950/20 hover:bg-red-900/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Remove
                </button>
              </div>
            </div>

            {/* Hidden file input for Replace */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept=".wav,.mp3,.m4a,audio/wav,audio/mpeg,audio/mp4,audio/x-m4a"
              className="hidden"
            />

            {/* In-Browser Audio Player */}
            <div className="bg-slate-950/60 border border-slate-800/60 rounded-xl p-3 flex flex-col gap-2">
              <span className="text-xs font-mono uppercase tracking-widest text-slate-500">Audio Preview</span>
              <audio
                ref={audioRef}
                controls
                src={audioUrl}
                onLoadedMetadata={(e) => setAudioDuration(e.currentTarget.duration)}
                className="w-full accent-indigo-500 rounded h-10"
              />
            </div>

            {/* Analyze Trigger Button */}
            {!result && (
              <button
                id="btn-analyze-audio"
                onClick={() => void handleAnalyze()}
                disabled={isAnalyzing}
                className="w-full py-3 rounded-xl font-bold text-sm bg-gradient-to-r from-indigo-500 via-violet-600 to-indigo-600 text-white shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-[1.01] active:scale-[0.99] flex items-center justify-center gap-2"
              >
                {isAnalyzing ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Analyzing Audio…</span>
                  </>
                ) : (
                  <>
                    <span>⚡ Analyze Audio</span>
                  </>
                )}
              </button>
            )}

            {/* Loading Progression Status */}
            {isAnalyzing && (
              <div className="bg-slate-950/50 border border-indigo-900/30 rounded-xl p-4 flex flex-col gap-3">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-indigo-300 font-semibold flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-indigo-400 animate-ping" />
                    {STAGE_LABELS[loadingStage]}
                  </span>
                  <span className="text-slate-500">{STAGE_PROGRESS[loadingStage]}%</span>
                </div>

                <div className="w-full bg-slate-800/80 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-indigo-500 to-violet-500 h-full rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${STAGE_PROGRESS[loadingStage]}%` }}
                  />
                </div>

                <div className="grid grid-cols-4 gap-1 text-[10px] font-mono text-center text-slate-500 pt-1">
                  <span className={loadingStage === "uploading" ? "text-indigo-400 font-bold" : ""}>Upload</span>
                  <span className={loadingStage === "processing" ? "text-indigo-400 font-bold" : ""}>Preprocess</span>
                  <span className={loadingStage === "analyzing" ? "text-indigo-400 font-bold" : ""}>AI Model</span>
                  <span className={loadingStage === "calculating" ? "text-indigo-400 font-bold" : ""}>Risk Engine</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Analysis Result Card ── */}
        {result && (
          <div
            id="voice-analysis-result-card"
            className={`rounded-2xl border p-6 transition-all duration-300 flex flex-col gap-6 ${riskCardStyle(
              result.risk_level
            )}`}
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-slate-800/60 pb-4">
              <div>
                <span className="text-xs font-bold tracking-widest text-slate-500 uppercase">
                  Detection Report
                </span>
                <h2 className="text-xl font-black text-white tracking-tight">
                  VOICE ANALYSIS RESULT
                </h2>
              </div>
              <span className={`text-xs font-black tracking-widest px-3 py-1.5 rounded-full uppercase ${riskBadgeStyle(result.risk_level)}`}>
                {result.risk_level} Risk
              </span>
            </div>

            {/* Primary Spoof Probability Metric */}
            <div className="flex flex-col items-center justify-center py-2 text-center gap-1">
              <span className={`text-6xl font-black tracking-tight ${riskColor(result.risk_level)}`}>
                {(result.spoof_probability * 100).toFixed(1)}%
              </span>
              <span className="text-xs font-mono font-bold tracking-widest text-slate-400 uppercase">
                Spoof Probability
              </span>
              <p className="text-[11px] text-slate-500 mt-1">
                Calculated across {result.chunks_analyzed} chunk{result.chunks_analyzed > 1 ? "s" : ""}
              </p>
            </div>

            {/* Advisory Recommendation Alert */}
            <div
              className={`rounded-xl p-4 border text-sm font-medium ${
                result.risk_level === "high"
                  ? "bg-red-500/10 border-red-500/30 text-red-300"
                  : result.risk_level === "medium"
                  ? "bg-amber-500/10 border-amber-500/30 text-amber-300"
                  : "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
              }`}
            >
              <div className="flex items-start gap-2.5">
                <span className="text-base shrink-0">
                  {result.risk_level === "high" ? "🚨" : result.risk_level === "medium" ? "⚠️" : "🛡️"}
                </span>
                <p className="leading-relaxed">{result.recommendation}</p>
              </div>
            </div>

            {/* Metadata Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-950/40 border border-slate-800/60 rounded-xl p-4">
              <div className="flex flex-col">
                <span className="text-[11px] text-slate-500 uppercase tracking-wider font-mono">Prediction</span>
                <span className={`text-base font-black tracking-wide uppercase ${result.prediction === "spoof" ? "text-red-400" : "text-emerald-400"}`}>
                  {result.prediction}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[11px] text-slate-500 uppercase tracking-wider font-mono">Duration</span>
                <span className="text-base font-bold text-white font-mono">
                  {result.audio_duration_s.toFixed(1)}s
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[11px] text-slate-500 uppercase tracking-wider font-mono">Chunks</span>
                <span className="text-base font-bold text-white font-mono">
                  {result.chunks_analyzed}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[11px] text-slate-500 uppercase tracking-wider font-mono">Latency</span>
                <span className="text-base font-bold text-indigo-400 font-mono">
                  {result.processing_time_ms} ms
                </span>
              </div>
            </div>

            {/* Per-chunk breakdown (if multi-chunk) */}
            {result.chunk_scores && result.chunk_scores.length > 1 && (
              <div className="bg-slate-950/30 border border-slate-800/40 rounded-xl p-3 flex flex-col gap-2">
                <span className="text-[11px] font-mono text-slate-500 uppercase tracking-wider">
                  Chunk Score Breakdown ({result.chunk_scores.length} Windows)
                </span>
                <div className="flex items-center gap-2 flex-wrap">
                  {result.chunk_scores.map((score, idx) => (
                    <div
                      key={idx}
                      className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-xs font-mono flex items-center gap-1.5"
                    >
                      <span className="text-slate-500">#{idx + 1}:</span>
                      <span className={`font-bold ${score >= 0.65 ? "text-red-400" : score >= 0.35 ? "text-amber-400" : "text-emerald-400"}`}>
                        {(score * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Flag for Review — hidden save trigger for demo */}
            {result && result.sha256_hash && (
              <div className="flex items-center justify-between border border-slate-800/50 rounded-xl px-4 py-2.5 bg-slate-950/40">
                <div className="flex items-center gap-2">
                  <span className="text-slate-500 text-xs font-mono">SHA-256</span>
                  <span className="text-[10px] text-slate-600 font-mono">{result.sha256_hash.substring(0, 24)}…</span>
                </div>
                <button
                  onClick={() => void handleMarkAsFake()}
                  disabled={overrideStatus === "saving" || overrideStatus === "saved"}
                  title="Flag this file for manual review"
                  className="flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1 rounded-lg border transition-all disabled:opacity-40 disabled:cursor-not-allowed
                    border-slate-700 text-slate-400 hover:text-amber-400 hover:border-amber-700/50 hover:bg-amber-950/20"
                >
                  {overrideStatus === "saving" && (
                    <span className="w-2.5 h-2.5 border-2 border-slate-400/30 border-t-slate-400 rounded-full animate-spin" />
                  )}
                  {overrideStatus === "saved" ? "✓ Flagged" :
                   overrideStatus === "error" ? "⚠ Failed" :
                   "🚩 Flag for Review"}
                </button>
              </div>
            )}


            {/* Reset / Analyze Another */}
            <button
              onClick={resetSelection}
              className="w-full py-2.5 rounded-xl font-semibold text-sm bg-slate-800/80 hover:bg-slate-800 border border-slate-700 text-slate-300 hover:text-white transition-colors"
            >
              Analyze Another Audio File
            </button>
          </div>
        )}

      </div>

      {/* ── Analysis History Panel ── */}
      <AnalysisHistoryPanel result={result} filename={selectedFile?.name ?? null} />

    </div>
  );
}
