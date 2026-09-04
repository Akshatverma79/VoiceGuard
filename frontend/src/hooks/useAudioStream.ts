/**
 * VoiceGuard — src/hooks/useAudioStream.ts
 * React hook: microphone capture + WebSocket streaming + state management.
 *
 * Responsibilities:
 *  - Request microphone permission
 *  - Set up Web Audio API context and AudioWorklet
 *  - Open WebSocket to /ws/audio
 *  - Forward PCM binary frames from AudioWorklet → WebSocket
 *  - Parse incoming server messages and update React state
 *  - Handle reconnection (max 2 retries on unexpected disconnect)
 *  - Clean up all resources on stopAnalysis() or unmount
 *
 * AudioWorklet fires every 128 samples (≈ 2.9 ms at 44 100 Hz).
 * We forward each block as a binary WebSocket frame immediately.
 * The server-side AudioBuffer accumulates until 4 s worth of audio,
 * then triggers AASIST inference.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { ServerMessage, DetectionMessage, PerfMessage } from "../types/detection";

const WS_URL = "ws://localhost:8000/ws/audio";
const MAX_RECONNECT_ATTEMPTS = 2;
const RECONNECT_DELAY_MS = 2000;

export type SessionStatus =
  | "idle"
  | "requesting_mic"
  | "connecting"
  | "listening"
  | "processing"
  | "waiting_for_speech"
  | "error"
  | "stopped";

export interface AudioStreamState {
  sessionStatus: SessionStatus;
  wsStatus: "disconnected" | "connecting" | "connected";
  micGranted: boolean;
  latestDetection: DetectionMessage | null;
  latestPerf: PerfMessage | null;
  statusMessage: string;
  errorMessage: string | null;
  scoreHistory: number[];
}

export interface AudioStreamActions {
  startAnalysis: () => Promise<void>;
  stopAnalysis: () => void;
}

export function useAudioStream(): AudioStreamState & AudioStreamActions {
  // ── State ──────────────────────────────────────────────────────────────
  const [sessionStatus, setSessionStatus] = useState<SessionStatus>("idle");
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected">("disconnected");
  const [micGranted, setMicGranted] = useState(false);
  const [latestDetection, setLatestDetection] = useState<DetectionMessage | null>(null);
  const [latestPerf, setLatestPerf] = useState<PerfMessage | null>(null);
  const [statusMessage, setStatusMessage] = useState("Ready");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [scoreHistory, setScoreHistory] = useState<number[]>([]);

  // ── Refs (non-reactive resources) ─────────────────────────────────────
  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const activeRef = useRef(false);  // guards against stale closures on cleanup

  // ── Cleanup helper ─────────────────────────────────────────────────────
  const cleanup = useCallback(() => {
    activeRef.current = false;

    // Stop AudioWorklet
    if (workletNodeRef.current) {
      workletNodeRef.current.port.postMessage({ command: "stop" });
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }

    // Stop microphone tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    // Close AudioContext
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      try {
        if (wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: "stop" }));
        }
        wsRef.current.close(1000, "User stopped");
      } catch (_) {}
      wsRef.current = null;
    }

    setMicGranted(false);
    setWsStatus("disconnected");
  }, []);

  // ── Start WebSocket ────────────────────────────────────────────────────
  const openWebSocket = useCallback((sampleRate: number) => {
    if (!activeRef.current) return;

    setWsStatus("connecting");
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      if (!activeRef.current) { ws.close(); return; }
      setWsStatus("connected");
      reconnectAttemptsRef.current = 0;
      // Send config first
      ws.send(JSON.stringify({ type: "config", sample_rate: sampleRate }));
    };

    ws.onmessage = (event) => {
      if (!activeRef.current) return;

      let msg: ServerMessage;
      try {
        msg = JSON.parse(event.data as string) as ServerMessage;
      } catch {
        return;
      }

      switch (msg.type) {
        case "status":
          if (msg.status === "connected" || msg.status === "listening") {
            setSessionStatus("listening");
            setStatusMessage("🎙️ Listening…");
          } else if (msg.status === "waiting_for_speech") {
            setSessionStatus("waiting_for_speech");
            setStatusMessage("🔇 Waiting for speech…");
          } else if (msg.status === "processing") {
            setSessionStatus("processing");
            setStatusMessage("⚙️ Analyzing…");
          } else if (msg.status === "timeout") {
            setSessionStatus("error");
            setStatusMessage("⏱️ Connection timed out");
          }
          break;

        case "detection":
          setSessionStatus("listening");
          setStatusMessage("🎙️ Listening…");
          setLatestDetection(msg);
          setScoreHistory(msg.history);
          break;

        case "perf":
          setLatestPerf(msg);
          break;

        case "error":
          setErrorMessage(msg.message);
          setStatusMessage(`⚠️ ${msg.message}`);
          break;
      }
    };

    ws.onerror = () => {
      setWsStatus("disconnected");
      setErrorMessage("WebSocket connection error");
    };

    ws.onclose = (ev) => {
      setWsStatus("disconnected");
      wsRef.current = null;

      if (!activeRef.current) return;

      // Unexpected close → attempt reconnect
      if (
        ev.code !== 1000 &&
        reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS
      ) {
        reconnectAttemptsRef.current += 1;
        setStatusMessage(
          `⚡ Reconnecting (${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})…`
        );
        setTimeout(() => {
          if (activeRef.current) openWebSocket(sampleRate);
        }, RECONNECT_DELAY_MS);
      } else {
        setSessionStatus("stopped");
        setStatusMessage("Connection closed. Click Start to retry.");
        cleanup();
      }
    };
  }, [cleanup]);

  // ── Start Analysis ─────────────────────────────────────────────────────
  const startAnalysis = useCallback(async () => {
    if (activeRef.current) return;

    setErrorMessage(null);
    setLatestDetection(null);
    setLatestPerf(null);
    setScoreHistory([]);
    activeRef.current = true;

    // Step 1: Request mic
    setSessionStatus("requesting_mic");
    setStatusMessage("Requesting microphone permission…");

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: { ideal: 16000 },  // hint only — browser may use native rate
        },
        video: false,
      });
    } catch (err: unknown) {
      const msg =
        err instanceof DOMException && err.name === "NotAllowedError"
          ? "Microphone permission denied. Please allow microphone access and try again."
          : `Microphone error: ${err instanceof Error ? err.message : String(err)}`;
      setErrorMessage(msg);
      setSessionStatus("error");
      setStatusMessage(msg);
      activeRef.current = false;
      return;
    }

    streamRef.current = stream;
    setMicGranted(true);

    // Step 2: Create AudioContext + AudioWorklet
    const audioCtx = new AudioContext();
    audioCtxRef.current = audioCtx;
    const nativeSampleRate = audioCtx.sampleRate;

    try {
      await audioCtx.audioWorklet.addModule("/audio-processor.js");
    } catch (err) {
      setErrorMessage(`Failed to load AudioWorklet: ${err}`);
      setSessionStatus("error");
      cleanup();
      return;
    }

    const source = audioCtx.createMediaStreamSource(stream);
    const workletNode = new AudioWorkletNode(audioCtx, "voiceguard-processor");
    workletNodeRef.current = workletNode;

    // Forward PCM blocks to WebSocket as binary
    workletNode.port.onmessage = (event) => {
      if (!activeRef.current) return;
      const { buffer } = event.data as { buffer: Float32Array };
      if (
        wsRef.current &&
        wsRef.current.readyState === WebSocket.OPEN &&
        buffer instanceof Float32Array
      ) {
        wsRef.current.send(buffer.buffer as ArrayBuffer);
      }
    };

    source.connect(workletNode);
    workletNode.connect(audioCtx.destination);  // required for worklet to fire

    // Step 3: Open WebSocket
    setSessionStatus("connecting");
    openWebSocket(nativeSampleRate);
  }, [openWebSocket, cleanup]);

  // ── Stop Analysis ──────────────────────────────────────────────────────
  const stopAnalysis = useCallback(() => {
    cleanup();
    setSessionStatus("stopped");
    setStatusMessage("Stopped. Click Start Analysis to run again.");
  }, [cleanup]);

  // ── Cleanup on unmount ─────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      activeRef.current = false;
      cleanup();
    };
  }, [cleanup]);

  return {
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
  };
}
