/**
 * VoiceGuard — src/App.tsx
 * Landing page for Phase 1.
 *
 * Displays:
 *   - VoiceGuard branding
 *   - Backend status (Connected / Disconnected)
 *
 * Polls GET /health every 5 seconds.
 */

import { useEffect, useState, useCallback } from "react";
import { checkHealth } from "./api/health";

type ConnectionStatus = "checking" | "connected" | "disconnected";

const POLL_INTERVAL_MS = 5000;

export default function App() {
  const [status, setStatus] = useState<ConnectionStatus>("checking");
  const [lastChecked, setLastChecked] = useState<string>("");

  const pingBackend = useCallback(async () => {
    const ok = await checkHealth();
    setStatus(ok ? "connected" : "disconnected");
    setLastChecked(new Date().toLocaleTimeString());
  }, []);

  useEffect(() => {
    // Initial check
    pingBackend();

    // Poll every 5 seconds
    const interval = setInterval(pingBackend, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [pingBackend]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 flex flex-col items-center justify-center relative overflow-hidden">
      {/* Ambient background orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl pointer-events-none animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-violet-600/10 rounded-full blur-3xl pointer-events-none animate-pulse" style={{ animationDelay: "1s" }} />

      {/* Main content */}
      <main className="relative z-10 flex flex-col items-center gap-8 px-6 text-center">
        {/* Shield icon */}
        <div className="flex items-center justify-center w-24 h-24 rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 shadow-2xl shadow-indigo-500/30 mb-2">
          <ShieldIcon className="w-14 h-14 text-white" />
        </div>

        {/* Brand */}
        <div className="space-y-3">
          <h1 className="text-6xl font-black tracking-tight text-white leading-none">
            Voice<span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-violet-400">Guard</span>
          </h1>
          <p className="text-xl text-slate-400 font-medium tracking-wide max-w-lg">
            AI-Powered Voice Deepfake Detection
          </p>
          <p className="text-sm text-slate-600 font-mono tracking-widest uppercase">
            SIH26104 · Phase 1
          </p>
        </div>

        {/* Status card */}
        <div className="mt-4 w-full max-w-sm">
          <div className="bg-slate-900/60 backdrop-blur-xl border border-slate-800/80 rounded-2xl p-6 shadow-xl">
            <p className="text-xs font-semibold tracking-widest text-slate-500 uppercase mb-4">
              Backend Status
            </p>

            <div className="flex items-center justify-center gap-3">
              <StatusDot status={status} />
              <StatusLabel status={status} />
            </div>

            {lastChecked && (
              <p className="mt-4 text-xs text-slate-600 text-center">
                Last checked: {lastChecked}
              </p>
            )}
          </div>
        </div>

        {/* Endpoint badge */}
        <div className="flex items-center gap-2 bg-slate-900/40 border border-slate-800 rounded-full px-4 py-2">
          <span className="text-xs font-mono text-slate-500">GET</span>
          <span className="text-xs font-mono text-indigo-400">http://localhost:8000/health</span>
        </div>
      </main>

      {/* Footer */}
      <footer className="absolute bottom-6 text-center text-slate-700 text-xs font-mono">
        VoiceGuard · Phase 1 · React + FastAPI + AASIST
      </footer>
    </div>
  );
}

/* ── Sub-components ──────────────────────────────────────────────────────── */

function StatusDot({ status }: { status: ConnectionStatus }) {
  const base = "w-3 h-3 rounded-full";

  if (status === "checking") {
    return (
      <span className={`${base} bg-amber-400 animate-pulse`} />
    );
  }
  if (status === "connected") {
    return (
      <span className="relative flex h-3 w-3">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
        <span className={`${base} relative bg-emerald-400`} />
      </span>
    );
  }
  return <span className={`${base} bg-red-500`} />;
}

function StatusLabel({ status }: { status: ConnectionStatus }) {
  if (status === "checking") {
    return <span className="text-amber-400 font-semibold text-lg">Checking…</span>;
  }
  if (status === "connected") {
    return <span className="text-emerald-400 font-semibold text-lg">Connected</span>;
  }
  return <span className="text-red-400 font-semibold text-lg">Disconnected</span>;
}

function ShieldIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  );
}
