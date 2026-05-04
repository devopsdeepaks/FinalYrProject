"use client";

import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle } from "lucide-react";
import ConfidenceGauge from "./ConfidenceGauge";

export interface AnalysisResult {
  is_fake: boolean;
  confidence: number;
  fake_probability: number;
  real_probability: number;
  face_detected: boolean;
  imageUrl: string;
  filename: string;
  timestamp: Date;
}

interface Props {
  result: AnalysisResult;
}

function StatRow({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-surface-border last:border-0">
      <span className="text-sm text-slate-400">{label}</span>
      <span
        className="text-sm font-mono font-semibold"
        style={color ? { color } : undefined}
      >
        {value}
      </span>
    </div>
  );
}

export default function ResultCard({ result }: Props) {
  const isFake = result.is_fake;
  const color = isFake ? "#e11d48" : "#16a34a";
  const colorLight = isFake ? "#ff4d6d" : "#4ade80";
  const glowClass = isFake ? "glow-fake" : "glow-real";
  const Icon = isFake ? AlertTriangle : CheckCircle;

  return (
    <motion.div
      className={`rounded-2xl border overflow-hidden ${glowClass}`}
      style={{ borderColor: `${color}44`, background: "#161b27" }}
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, type: "spring", stiffness: 200, damping: 20 }}
    >
      {/* Header banner */}
      <div
        className="px-5 py-3 flex items-center gap-2"
        style={{ background: `${color}18`, borderBottom: `1px solid ${color}33` }}
      >
        <Icon size={16} style={{ color: colorLight }} />
        <span className="text-sm font-semibold" style={{ color: colorLight }}>
          {isFake ? "Deepfake Detected" : "Authentic Image"}
        </span>
        <span className="ml-auto text-xs text-slate-500 font-mono">
          {result.timestamp.toLocaleTimeString()}
        </span>
      </div>

      <div className="p-5 grid grid-cols-1 sm:grid-cols-2 gap-6">
        {/* Thumbnail */}
        <div className="relative aspect-square rounded-xl overflow-hidden border border-surface-border">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.imageUrl}
            alt={result.filename}
            className="w-full h-full object-cover"
          />
          <div
            className="absolute inset-0 opacity-20"
            style={{
              background: `radial-gradient(circle at center, ${color}, transparent 70%)`,
            }}
          />
        </div>

        {/* Stats */}
        <div className="flex flex-col gap-4">
          <ConfidenceGauge value={result.confidence} isFake={isFake} />

          <div className="rounded-xl border border-surface-border p-3 space-y-0">
            <StatRow
              label="Fake probability"
              value={`${(result.fake_probability * 100).toFixed(1)}%`}
              color="#ff4d6d"
            />
            <StatRow
              label="Real probability"
              value={`${(result.real_probability * 100).toFixed(1)}%`}
              color="#4ade80"
            />
            <StatRow
              label="Face detected"
              value={result.face_detected ? "Yes" : "No"}
              color={result.face_detected ? "#a5b4fc" : "#94a3b8"}
            />
            <StatRow label="File" value={result.filename} />
          </div>
        </div>
      </div>
    </motion.div>
  );
}
