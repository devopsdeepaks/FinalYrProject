"use client";

import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, CheckCircle, Trash2 } from "lucide-react";
import type { AnalysisResult } from "./ResultCard";

interface Props {
  items: AnalysisResult[];
  onSelect: (item: AnalysisResult) => void;
  onClear: () => void;
  selected: AnalysisResult | null;
}

export default function HistoryList({ items, onSelect, onClear, selected }: Props) {
  if (items.length === 0) return null;

  return (
    <div className="rounded-2xl border border-surface-border bg-surface-card overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-surface-border">
        <span className="text-sm font-semibold text-slate-300">History</span>
        <button
          onClick={onClear}
          className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-fake transition-colors"
        >
          <Trash2 size={12} />
          Clear
        </button>
      </div>

      <ul className="divide-y divide-surface-border max-h-64 overflow-y-auto">
        <AnimatePresence initial={false}>
          {items.map((item, i) => {
            const isSelected = selected?.timestamp === item.timestamp;
            const isFake = item.is_fake;
            const color = isFake ? "#e11d48" : "#16a34a";
            const Icon = isFake ? AlertTriangle : CheckCircle;

            return (
              <motion.li
                key={item.timestamp.toISOString()}
                layout
                initial={{ opacity: 0, x: -16 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 16 }}
                transition={{ delay: i * 0.04 }}
                onClick={() => onSelect(item)}
                className={`flex items-center gap-3 px-4 py-3 cursor-pointer transition-colors
                  ${isSelected ? "bg-surface-muted" : "hover:bg-surface-border/40"}`}
              >
                {/* Thumb */}
                <div className="w-10 h-10 rounded-lg overflow-hidden shrink-0 border border-surface-border">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={item.imageUrl}
                    alt={item.filename}
                    className="w-full h-full object-cover"
                  />
                </div>

                <div className="flex-1 min-w-0">
                  <p className="text-sm text-slate-300 truncate">{item.filename}</p>
                  <p className="text-xs text-slate-500">
                    {item.timestamp.toLocaleTimeString()}
                  </p>
                </div>

                <div className="flex flex-col items-end gap-0.5 shrink-0">
                  <Icon size={14} style={{ color }} />
                  <span className="text-xs font-mono" style={{ color }}>
                    {Math.round(item.confidence * 100)}%
                  </span>
                </div>
              </motion.li>
            );
          })}
        </AnimatePresence>
      </ul>
    </div>
  );
}
