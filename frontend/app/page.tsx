"use client";

import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const EMOTION_EMOJI: Record<string, string> = {
  angry: "😠",
  calm: "😌",
  disgust: "🤢",
  fearful: "😨",
  happy: "😊",
  neutral: "😐",
  sad: "😢",
  surprised: "😲",
};

const EMOTION_COLOR: Record<string, string> = {
  angry: "bg-red-500",
  calm: "bg-blue-400",
  disgust: "bg-green-600",
  fearful: "bg-purple-500",
  happy: "bg-yellow-400",
  neutral: "bg-gray-400",
  sad: "bg-indigo-500",
  surprised: "bg-pink-500",
};

type PredictResponse = {
  emotion: string;
  confidence: number | null;
  all_scores: Record<string, number>;
  duration_seconds: number;
};

export default function Home() {
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  async function blobToWav(blob: Blob): Promise<Blob> {
    const buf = await blob.arrayBuffer();
    const AudioCtx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new AudioCtx();
    const audioBuf = await ctx.decodeAudioData(buf.slice(0));
    await ctx.close();

    const numCh = audioBuf.numberOfChannels;
    const sr = audioBuf.sampleRate;
    const samples = audioBuf.length;
    const dataLen = samples * numCh * 2;
    const out = new ArrayBuffer(44 + dataLen);
    const v = new DataView(out);

    const writeStr = (off: number, s: string) => {
      for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
    };

    writeStr(0, "RIFF");
    v.setUint32(4, 36 + dataLen, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    v.setUint32(16, 16, true);
    v.setUint16(20, 1, true);
    v.setUint16(22, numCh, true);
    v.setUint32(24, sr, true);
    v.setUint32(28, sr * numCh * 2, true);
    v.setUint16(32, numCh * 2, true);
    v.setUint16(34, 16, true);
    writeStr(36, "data");
    v.setUint32(40, dataLen, true);

    const channels: Float32Array[] = [];
    for (let c = 0; c < numCh; c++) channels.push(audioBuf.getChannelData(c));

    let off = 44;
    for (let i = 0; i < samples; i++) {
      for (let c = 0; c < numCh; c++) {
        const s = Math.max(-1, Math.min(1, channels[c][i]));
        v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        off += 2;
      }
    }
    return new Blob([out], { type: "audio/wav" });
  }

  async function predict(blob: Blob, name: string) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const wav = await blobToWav(blob);
      const wavName = name.replace(/\.[^.]+$/, "") + ".wav";
      const fd = new FormData();
      fd.append("file", wav, wavName);
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: fd });
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(`API ${res.status}: ${msg}`);
      }
      const json = (await res.json()) as PredictResponse;
      setResult(json);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function startRecording() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mr.mimeType || "audio/webm" });
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        setFilename("recording.webm");
        stream.getTracks().forEach((t) => t.stop());
        predict(blob, "recording.webm");
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setRecording(true);
    } catch (e) {
      setError(`Microphone error: ${e instanceof Error ? e.message : e}`);
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    setFilename(file.name);
    predict(file, file.name);
  }

  const sortedScores = result
    ? Object.entries(result.all_scores).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <main className="min-h-screen bg-slate-900 text-white">
      <div className="mx-auto max-w-2xl px-6 py-12">
        <header className="mb-10">
          <h1 className="text-4xl font-bold tracking-tight">Speech Emotion Recognition</h1>
          <p className="mt-2 text-slate-400">
            Upload an audio file or record from your mic. The model classifies the dominant emotion across 8 categories.
          </p>
        </header>

        <section className="grid gap-4 sm:grid-cols-2 mb-8">
          <button
            onClick={recording ? stopRecording : startRecording}
            disabled={loading}
            className={`rounded-lg px-6 py-4 font-semibold transition ${
              recording
                ? "bg-red-600 hover:bg-red-500 animate-pulse"
                : "bg-blue-600 hover:bg-blue-500"
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {recording ? "■ Stop Recording" : "🎤 Record from Mic"}
          </button>

          <label className="rounded-lg px-6 py-4 font-semibold bg-slate-700 hover:bg-slate-600 text-center cursor-pointer transition">
            📁 Upload Audio
            <input
              type="file"
              accept=".wav,.mp3,.ogg,.flac,.m4a,.webm,audio/*"
              onChange={handleFile}
              disabled={loading || recording}
              className="hidden"
            />
          </label>
        </section>

        {audioUrl && (
          <section className="mb-8 rounded-lg bg-slate-800 p-4">
            <p className="text-sm text-slate-400 mb-2">{filename}</p>
            <audio src={audioUrl} controls className="w-full" />
          </section>
        )}

        {loading && (
          <div className="rounded-lg bg-slate-800 p-6 text-center text-slate-300">
            Analyzing audio…
          </div>
        )}

        {error && (
          <div className="rounded-lg bg-red-950 border border-red-800 p-4 text-red-200">
            {error}
          </div>
        )}

        {result && !loading && (
          <section className="rounded-lg bg-slate-800 p-6">
            <div className="mb-6 text-center">
              <div className="text-7xl mb-2">{EMOTION_EMOJI[result.emotion] ?? "❓"}</div>
              <div className="text-3xl font-bold capitalize">{result.emotion}</div>
              {result.confidence !== null && (
                <div className="text-slate-400 mt-1">
                  {(result.confidence * 100).toFixed(1)}% confidence · {result.duration_seconds}s clip
                </div>
              )}
            </div>

            <div className="space-y-2">
              {sortedScores.map(([emo, score]) => (
                <div key={emo}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="capitalize">
                      {EMOTION_EMOJI[emo]} {emo}
                    </span>
                    <span className="text-slate-400 font-mono">
                      {(score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-700 overflow-hidden">
                    <div
                      className={`h-full ${EMOTION_COLOR[emo] ?? "bg-slate-500"} transition-all duration-500`}
                      style={{ width: `${score * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        <footer className="mt-12 text-center text-xs text-slate-500">
          API: <code>{API_URL}</code>
        </footer>
      </div>
    </main>
  );
}
