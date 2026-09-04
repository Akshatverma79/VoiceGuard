/**
 * VoiceGuard — src/App.tsx
 * Root application component — Phase 1 + Phase 2 + Phase 3.
 *
 * Tab-based navigation (no react-router dependency):
 *   "home" → Landing page with backend health status
 *   "live" → Real-time detection page (Phase 3)
 */

import { useEffect, useState, useCallback } from "react";
import { checkHealth } from "./api/health";
import Live from "./pages/Live";

type Tab = "home" | "live";
type ConnectionStatus = "checking" | "connected" | "disconnected";

const POLL_INTERVAL_MS = 5000;

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("home");
  const [status, setStatus] = useState<ConnectionStatus>("checking");
  const [lastChecked, setLastChecked] = useState<string>("");

  const pingBackend = useCallback(async () => {
    const ok = await checkHealth();
    setStatus(ok ? "connected" : "disconnected");
    setLastChecked(new Date().toLocaleTimeString());
  }, []);

  useEffect(() => {
    pingBackend();
    const interval = setInterval(pingBackend, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [pingBackend]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 flex flex-col">

      {/* ── Top nav ── */}
      <nav className="flex items-center justify-between px-6 py-4 border-b border-slate-800/60 backdrop-blur-sm sticky top-0 z-50 bg-slate-950/80">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 shadow-lg shadow-indigo-500/20">
            <ShieldIcon className="w-5 h-5 text-white" />
          </div>
          <span className="text-white font-black tracking-tight">
            Voice<span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-violet-400">Guard</span>
          </span>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-1 bg-slate-900/60 border border-slate-800 rounded-xl p-1">
          <TabBtn
            id="tab-home"
            label="Home"
            active={activeTab === "home"}
            onClick={() => setActiveTab("home")}
          />
          <TabBtn
            id="tab-live"
            label="🎙️ Live Detection"
            active={activeTab === "live"}
            onClick={() => setActiveTab("live")}
          />
        </div>

        {/* Backend status pill */}
        <div className="flex items-center gap-2 bg-slate-900/60 border border-slate-800 rounded-full px-3 py-1.5">
          <StatusDot status={status} />
          <span className="text-xs text-slate-400 font-mono">
            {status === "connected" ? "API Online" : status === "checking" ? "Checking…" : "API Offline"}
          </span>
        </div>
      </nav>

      {/* ── Page content ── */}
      <main className="flex-1">
        {activeTab === "home" ? (
          <HomePage status={status} lastChecked={lastChecked} />
        ) : (
          <Live />
        )}
      </main>
    </div>
  );
}

/* ── Home page ───────────────────────────────────────────────────────────── */

function HomePage({
  status,
  lastChecked,
}: {
  status: ConnectionStatus;
  lastChecked: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] relative overflow-hidden px-6 text-center gap-8">
      {/* Ambient orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl pointer-events-none animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-violet-600/10 rounded-full blur-3xl pointer-events-none animate-pulse" style={{ animationDelay: "1s" }} />

      {/* Shield icon */}
      <div className="relative z-10 flex items-center justify-center w-28 h-28 rounded-3xl bg-gradient-to-br from-indigo-500 to-violet-600 shadow-2xl shadow-indigo-500/30">
        <ShieldIcon className="w-16 h-16 text-white" />
      </div>

      {/* Brand */}
      <div className="relative z-10 space-y-3">
        <h1 className="text-6xl font-black tracking-tight text-white leading-none">
          Voice<span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-violet-400">Guard</span>
        </h1>
        <p className="text-xl text-slate-400 font-medium tracking-wide max-w-lg">
          AI-Powered Voice Deepfake Detection
        </p>
        <p className="text-sm text-slate-600 font-mono tracking-widest uppercase">
          SIH26104 · Phase 3 · React + FastAPI + AASIST
        </p>
      </div>

      {/* Feature badges */}
      <div className="relative z-10 flex flex-wrap justify-center gap-3">
        {["Real-time WebSocket", "Browser Microphone", "AASIST Inference", "Rolling Risk Engine"].map((f) => (
          <span
            key={f}
            className="text-xs font-medium text-slate-400 bg-slate-900/60 border border-slate-800 rounded-full px-3 py-1"
          >
            {f}
          </span>
        ))}
      </div>

      {/* Status card */}
      <div className="relative z-10 w-full max-w-sm">
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

      {/* CTA */}
      <div className="relative z-10">
        <p className="text-slate-500 text-sm">
          Click{" "}
          <span className="text-indigo-400 font-semibold">🎙️ Live Detection</span>{" "}
          in the navigation to start real-time analysis.
        </p>
      </div>
    </div>
  );
}

/* ── Sub-components ──────────────────────────────────────────────────────── */

function TabBtn({
  id, label, active, onClick,
}: {
  id: string;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      id={id}
      onClick={onClick}
      className={`text-sm font-semibold px-4 py-1.5 rounded-lg transition-all duration-200 ${
        active
          ? "bg-gradient-to-r from-indigo-500 to-violet-600 text-white shadow-md"
          : "text-slate-400 hover:text-white"
      }`}
    >
      {label}
    </button>
  );
}

function StatusDot({ status }: { status: ConnectionStatus }) {
  if (status === "checking") {
    return <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse inline-block" />;
  }
  if (status === "connected") {
    return (
      <span className="relative flex h-2.5 w-2.5">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
        <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-400" />
      </span>
    );
  }
  return <span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block" />;
}

function StatusLabel({ status }: { status: ConnectionStatus }) {
  if (status === "checking") return <span className="text-amber-400 font-semibold text-lg">Checking…</span>;
  if (status === "connected") return <span className="text-emerald-400 font-semibold text-lg">Connected</span>;
  return <span className="text-red-400 font-semibold text-lg">Disconnected</span>;
}

function ShieldIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  );
}
