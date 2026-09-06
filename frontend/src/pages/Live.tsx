/**
 * VoiceVeritas — src/pages/Live.tsx
 * Real-time live detection page (Phase 3).
 *
 * Layout:
 *  - Header: VoiceVeritas branding + phase badge
 *  - Control row: Start / Stop buttons, status indicators
 *  - Primary display: risk level ring + spoof probability
 *  - Score sparkline: recent chunk scores
 *  - Info grid: connection status, VAD status, inference timing
 *  - Error banner
 */

import { useAudioStream, type SessionStatus } from "../hooks/useAudioStream";
import type { RiskLevel } from "../types/detection";

// ── Colour helpers ──────────────────────────────────────────────────────────

function riskColor(level: RiskLevel): string {
  switch (level) {
    case "high": return "text-red-400";
    case "medium": return "text-amber-400";
    case "low": return "text-emerald-400";
    default: return "text-slate-500";
  }
}

function riskBg(level: RiskLevel): string {
  switch (level) {
    case "high": return "bg-red-500/10 border-red-500/30";
    case "medium": return "bg-amber-500/10 border-amber-500/30";
    case "low": return "bg-emerald-500/10 border-emerald-500/30";
    default: return "bg-slate-800/40 border-slate-700/40";
  }
}

function riskGlow(level: RiskLevel): string {
  switch (level) {
    case "high": return "shadow-red-500/20";
    case "medium": return "shadow-amber-500/20";
    case "low": return "shadow-emerald-500/20";
    default: return "shadow-slate-700/10";
  }
}

function riskLabel(level: RiskLevel): string {
  switch (level) {
    case "high": return "HIGH RISK";
    case "medium": return "MEDIUM RISK";
    case "low": return "LOW RISK";
    default: return "ANALYZING…";
  }
}

function predictionColor(p: "real" | "spoof" | null): string {
  if (p === "real") return "text-emerald-400";
  if (p === "spoof") return "text-red-400";
  return "text-slate-500";
}

// ── Sparkline ───────────────────────────────────────────────────────────────

function Sparkline({ scores }: { scores: number[] }) {
  if (scores.length < 2) {
    return (
      <div className="h-12 flex items-center justify-center text-slate-600 text-xs">
        Waiting for data…
      </div>
    );
  }

  const W = 280;
  const H = 48;
  const points = scores.map((s, i) => {
    const x = (i / (scores.length - 1)) * W;
    const y = H - s * H;
    return `${x},${y}`;
  });
  const polyline = points.join(" ");

  // Color last point by value
  const last = scores[scores.length - 1];
  const dotColor = last > 0.65 ? "#f87171" : last > 0.35 ? "#fbbf24" : "#34d399";

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="overflow-visible">
      {/* threshold lines */}
      <line x1={0} y1={H * (1 - 0.65)} x2={W} y2={H * (1 - 0.65)}
        stroke="#f87171" strokeOpacity={0.2} strokeWidth={1} strokeDasharray="3 3" />
      <line x1={0} y1={H * (1 - 0.35)} x2={W} y2={H * (1 - 0.35)}
        stroke="#fbbf24" strokeOpacity={0.2} strokeWidth={1} strokeDasharray="3 3" />

      <polyline
        points={polyline}
        fill="none"
        stroke="#818cf8"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Last point dot */}
      <circle
        cx={Number(points[points.length - 1].split(",")[0])}
        cy={Number(points[points.length - 1].split(",")[1])}
        r={3}
        fill={dotColor}
      />
    </svg>
  );
}

// ── Status dot ──────────────────────────────────────────────────────────────

function StatusDot({ active }: { active: boolean }) {
  if (active) {
    return (
      <span className="relative flex h-2.5 w-2.5">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
        <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-400" />
      </span>
    );
  }
  return <span className="inline-flex h-2.5 w-2.5 rounded-full bg-slate-600" />;
}

// ── Session status badge ────────────────────────────────────────────────────

function SessionBadge({ status }: { status: SessionStatus }) {
  const map: Record<SessionStatus, { label: string; cls: string }> = {
    idle: { label: "IDLE", cls: "bg-slate-800 text-slate-400" },
    requesting_mic: { label: "REQUESTING MIC", cls: "bg-amber-500/20 text-amber-400" },
    connecting: { label: "CONNECTING", cls: "bg-indigo-500/20 text-indigo-400 animate-pulse" },
    listening: { label: "LIVE", cls: "bg-emerald-500/20 text-emerald-400" },
    processing: { label: "ANALYZING", cls: "bg-violet-500/20 text-violet-400 animate-pulse" },
    waiting_for_speech: { label: "WAITING", cls: "bg-slate-700 text-slate-400" },
    error: { label: "ERROR", cls: "bg-red-500/20 text-red-400" },
    stopped: { label: "STOPPED", cls: "bg-slate-800 text-slate-500" },
  };
  const { label, cls } = map[status];
  return (
    <span className={`text-xs font-bold tracking-widest px-2.5 py-1 rounded-full ${cls}`}>
      {label}
    </span>
  );
}

// ── Main component ──────────────────────────────────────────────────────────

export default function Live() {
  const {
    sessionStatus,
    wsStatus,
    micGranted,
    latestDetection,
    latestPerf,
    statusMessage,
    errorMessage,
    scoreHistory,
    startAnalysis,
    stopAnalysis,
  } = useAudioStream();

  const isActive =
    sessionStatus === "listening" ||
    sessionStatus === "processing" ||
    sessionStatus === "waiting_for_speech" ||
    sessionStatus === "connecting";

  const riskLvl: RiskLevel = latestDetection?.risk_level ?? "unknown";
  const spoofProb = latestDetection?.spoof_probability ?? null;
  const prediction = latestDetection?.prediction ?? null;
  const riskScore = latestDetection?.risk_score ?? null;

  // Format probability as percentage
  const probPct = spoofProb !== null ? `${(spoofProb * 100).toFixed(1)}%` : "—";
  const riskPct = riskScore !== null ? `${(riskScore * 100).toFixed(1)}%` : "—";

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 p-6 flex flex-col gap-6 items-center">

      {/* ── Header ── */}
      <div className="flex flex-col items-center gap-2 pt-4">
        <h1 className="text-3xl font-black tracking-tight text-white">
          Voice<span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-violet-400">Veritas</span>
          <span className="text-slate-500 text-xl font-medium ml-3">Live Detection</span>
        </h1>
        <p className="text-slate-500 text-sm font-mono">SIH26104 · Real-Time Analysis</p>
      </div>

      {/* ── Control row ── */}
      <div className="flex items-center gap-4 flex-wrap justify-center">
        <button
          id="btn-start-analysis"
          onClick={() => void startAnalysis()}
          disabled={isActive}
          className="px-6 py-2.5 rounded-xl font-semibold text-sm bg-gradient-to-r from-indigo-500 to-violet-600 text-white shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/40 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105 active:scale-95"
        >
          🎙️ Start Analysis
        </button>

        <button
          id="btn-stop-analysis"
          onClick={stopAnalysis}
          disabled={!isActive}
          className="px-6 py-2.5 rounded-xl font-semibold text-sm bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105 active:scale-95"
        >
          ⏹ Stop Analysis
        </button>

        <SessionBadge status={sessionStatus} />
      </div>

      {/* ── Error banner ── */}
      {errorMessage && (
        <div className="w-full max-w-xl bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400 text-sm text-center">
          ⚠️ {errorMessage}
        </div>
      )}

      {/* ── Status message ── */}
      <p className="text-slate-400 text-sm text-center">{statusMessage}</p>

      {/* ── Main cards grid ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 w-full max-w-2xl">

        {/* Primary risk card */}
        <div className={`col-span-1 md:col-span-2 rounded-2xl border p-6 shadow-xl ${riskBg(riskLvl)} ${riskGlow(riskLvl)}`}>
          <div className="flex items-center justify-between mb-4">
            <p className="text-xs font-bold tracking-widest text-slate-500 uppercase">Risk Level</p>
            <span className={`text-xs font-bold tracking-widest ${riskColor(riskLvl)}`}>
              {riskLabel(riskLvl)}
            </span>
          </div>

          <div className="flex items-center justify-around gap-6 flex-wrap">
            {/* Spoof probability */}
            <div className="flex flex-col items-center gap-1">
              <span className="text-5xl font-black tracking-tight text-white">{probPct}</span>
              <span className="text-xs text-slate-500 uppercase tracking-widest">
                {sessionStatus === "waiting_for_speech" && latestDetection ? "Average Spoof Probability" : "Spoof Probability"}
              </span>
            </div>

            {/* Divider */}
            <div className="h-16 w-px bg-slate-700/50 hidden sm:block" />

            {/* Prediction */}
            <div className="flex flex-col items-center gap-1">
              <span className={`text-3xl font-black uppercase tracking-widest ${predictionColor(prediction)}`}>
                {prediction ?? "—"}
              </span>
              <span className="text-xs text-slate-500 uppercase tracking-widest">Prediction</span>
            </div>

            {/* Divider */}
            <div className="h-16 w-px bg-slate-700/50 hidden sm:block" />

            {/* Risk score (EWMA) */}
            <div className="flex flex-col items-center gap-1">
              <span className={`text-3xl font-black ${riskColor(riskLvl)}`}>{riskPct}</span>
              <span className="text-xs text-slate-500 uppercase tracking-widest">Risk Score</span>
            </div>
          </div>
        </div>

        {/* Score sparkline */}
        <div className="col-span-1 md:col-span-2 bg-slate-900/60 backdrop-blur-xl border border-slate-800/60 rounded-2xl p-5 shadow-xl">
          <p className="text-xs font-bold tracking-widest text-slate-500 uppercase mb-3">Score History</p>
          <div className="flex items-center justify-center">
            <Sparkline scores={scoreHistory} />
          </div>
          <div className="flex justify-between mt-2 text-xs text-slate-600 font-mono">
            <span>Older</span>
            <span className="text-amber-600/60">─ 0.35 ─ 0.65 ─</span>
            <span>Latest</span>
          </div>
        </div>

        {/* Connection status */}
        <div className="bg-slate-900/60 backdrop-blur-xl border border-slate-800/60 rounded-2xl p-5 shadow-xl">
          <p className="text-xs font-bold tracking-widest text-slate-500 uppercase mb-4">Connection</p>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">WebSocket</span>
              <div className="flex items-center gap-2">
                <StatusDot active={wsStatus === "connected"} />
                <span className="text-xs text-slate-400 capitalize">{wsStatus}</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">Microphone</span>
              <div className="flex items-center gap-2">
                <StatusDot active={micGranted} />
                <span className="text-xs text-slate-400">{micGranted ? "Active" : "Off"}</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">Chunks seen</span>
              <span className="text-xs text-slate-300 font-mono">
                {latestDetection?.chunks_analyzed ?? 0}
              </span>
            </div>
          </div>
        </div>

        {/* Perf card */}
        <div className="bg-slate-900/60 backdrop-blur-xl border border-slate-800/60 rounded-2xl p-5 shadow-xl">
          <p className="text-xs font-bold tracking-widest text-slate-500 uppercase mb-4">Latency (last chunk)</p>
          {latestPerf ? (
            <div className="space-y-3">
              <Row label="Buffer" val={`${latestPerf.buffer_s.toFixed(2)} s`} />
              <Row label="Preprocess" val={`${latestPerf.preprocess_ms} ms`} />
              <Row label="AI Inference" val={`${latestPerf.inference_ms} ms`} />
              <Row label="Total" val={`${latestPerf.total_ms} ms`} highlight />
            </div>
          ) : (
            <p className="text-slate-600 text-sm text-center mt-4">No data yet</p>
          )}
        </div>

      </div>

      {/* ── Limitations notice ── */}
      <div className="w-full max-w-2xl bg-slate-900/40 border border-slate-800/40 rounded-xl p-4 text-slate-600 text-xs text-center">
        ⚠️ Thresholds (LOW &lt; 35%, MEDIUM 35–65%, HIGH ≥ 65%) are starting values and require calibration.
        Results are from real AI model inference, not fabricated values.
      </div>
    </div>
  );
}

function Row({ label, val, highlight }: { label: string; val: string; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-slate-400">{label}</span>
      <span className={`text-xs font-mono ${highlight ? "text-indigo-400 font-bold" : "text-slate-300"}`}>
        {val}
      </span>
    </div>
  );
}
