// AudioWorklet running inside the AudioContext: takes 128-sample (or so)
// frames at the context's sample rate, accumulates them, and emits 1024-sample
// chunks of int16 PCM to the main thread. The main thread forwards them as
// binary WebSocket frames.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(0);
    this._chunkSize = 1024;  // ~64ms at 16kHz
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];

    // Append to internal buffer.
    const merged = new Float32Array(this._buf.length + ch.length);
    merged.set(this._buf, 0);
    merged.set(ch, this._buf.length);
    this._buf = merged;

    // Flush in chunks.
    while (this._buf.length >= this._chunkSize) {
      const slice = this._buf.subarray(0, this._chunkSize);
      this._buf = this._buf.slice(this._chunkSize);

      const int16 = new Int16Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        let s = Math.max(-1, Math.min(1, slice[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      // Transfer the buffer to avoid a copy.
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }

    return true;
  }
}

registerProcessor('capture-processor', CaptureProcessor);
