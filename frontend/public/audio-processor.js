/**
 * VoiceVeritas — public/audio-processor.js
 * AudioWorklet processor — runs in the browser's audio rendering thread.
 *
 * Receives 128-sample float32 PCM blocks from the microphone and posts
 * them to the main thread as a binary ArrayBuffer (Float32Array wire format).
 *
 * IMPORTANT: This file must be served as a static asset (not bundled by Vite)
 * because AudioWorklet.addModule() requires a URL, not an imported module.
 * Place it in frontend/public/ so Vite serves it at /audio-processor.js.
 *
 * Message protocol (processor → main thread):
 *   { buffer: Float32Array }   — one 128-sample block of mono PCM
 *
 * Control messages (main thread → processor):
 *   { command: "stop" }        — stop posting audio
 */

class VoiceVeritasProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._active = true;

    this.port.onmessage = (event) => {
      if (event.data?.command === "stop") {
        this._active = false;
      }
    };
  }

  /**
   * Called by the browser's audio engine for each 128-sample render quantum.
   * inputs[0][0] is the first (mono) channel of the first input node.
   */
  process(inputs) {
    if (!this._active) return false; // returning false terminates the processor

    const channel = inputs?.[0]?.[0];
    if (!channel || channel.length === 0) return true;

    // Post a copy (not a transfer) so the audio engine can reuse the buffer
    this.port.postMessage({ buffer: channel.slice() });

    return true; // keep processor alive
  }
}

registerProcessor("voiceveritas-processor", VoiceVeritasProcessor);
