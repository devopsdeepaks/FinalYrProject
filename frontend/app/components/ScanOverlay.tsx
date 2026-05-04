"use client";

import { motion } from "framer-motion";

interface Props {
  imageUrl: string;
}

export default function ScanOverlay({ imageUrl }: Props) {
  return (
    <div className="relative w-full aspect-square overflow-hidden rounded-xl border border-brand-500/30 bg-surface-card">
      {/* Image */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imageUrl}
        alt="Analysing"
        className="w-full h-full object-cover"
      />

      {/* Dark overlay */}
      <div className="absolute inset-0 bg-surface/40" />

      {/* Scanning line */}
      <motion.div
        className="absolute left-0 w-full h-0.5"
        style={{
          background:
            "linear-gradient(90deg, transparent, #4f6ef7, #a5b4fc, #4f6ef7, transparent)",
          boxShadow: "0 0 12px 4px rgba(79,110,247,0.6)",
        }}
        initial={{ top: "0%" }}
        animate={{ top: "100%" }}
        transition={{ duration: 1.8, repeat: Infinity, ease: "linear" }}
      />

      {/* Corner brackets */}
      {[
        "top-3 left-3 border-t border-l",
        "top-3 right-3 border-t border-r",
        "bottom-3 left-3 border-b border-l",
        "bottom-3 right-3 border-b border-r",
      ].map((cls) => (
        <div
          key={cls}
          className={`absolute w-5 h-5 border-brand-500 ${cls}`}
        />
      ))}

      {/* Scanning label */}
      <div className="absolute bottom-3 left-0 right-0 flex justify-center">
        <motion.span
          className="text-xs font-mono text-brand-500 tracking-widest"
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 1.2, repeat: Infinity }}
        >
          ANALYSING...
        </motion.span>
      </div>
    </div>
  );
}
