"use client";

import { motion } from "framer-motion";

interface Props {
  value: number; // 0–1
  isFake: boolean;
}

export default function ConfidenceGauge({ value, isFake }: Props) {
  const radius = 64;
  const stroke = 8;
  const normalised = radius - stroke / 2;
  const circumference = 2 * Math.PI * normalised;
  const progress = circumference * (1 - value);

  const color = isFake ? "#e11d48" : "#16a34a";
  const colorLight = isFake ? "#ff4d6d" : "#4ade80";
  const label = isFake ? "FAKE" : "REAL";
  const percent = Math.round(value * 100);

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative w-40 h-40">
        {/* Background track */}
        <svg className="w-full h-full -rotate-90" viewBox="0 0 144 144">
          <circle
            cx="72"
            cy="72"
            r={normalised}
            fill="none"
            stroke="#1e2535"
            strokeWidth={stroke}
          />
          <motion.circle
            cx="72"
            cy="72"
            r={normalised}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: progress }}
            transition={{ duration: 1.2, ease: "easeOut" }}
            style={{ filter: `drop-shadow(0 0 6px ${color})` }}
          />
        </svg>

        {/* Centre label */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            className="text-4xl font-bold font-mono tabular-nums"
            style={{ color: colorLight }}
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5, duration: 0.5, type: "spring" }}
          >
            {percent}%
          </motion.span>
          <span className="text-xs text-slate-400 mt-0.5">confidence</span>
        </div>
      </div>

      {/* Verdict badge */}
      <motion.div
        className="px-6 py-1.5 rounded-full text-sm font-bold tracking-widest uppercase"
        style={{
          background: `${color}22`,
          border: `1px solid ${color}66`,
          color: colorLight,
        }}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
      >
        {label}
      </motion.div>
    </div>
  );
}
