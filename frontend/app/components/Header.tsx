"use client";

import { motion } from "framer-motion";
import { Shield, Activity } from "lucide-react";

export default function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-surface-border bg-surface/80 backdrop-blur-md">
      <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
        {/* Logo */}
        <motion.div
          className="flex items-center gap-2.5"
          initial={{ opacity: 0, x: -16 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <div className="p-1.5 rounded-lg bg-brand-500/15 border border-brand-500/30">
            <Shield size={18} className="text-brand-500" />
          </div>
          <span className="font-bold text-lg tracking-tight">
            Deep<span className="text-brand-500">Scan</span>
          </span>
        </motion.div>

        {/* Status pill */}
        <motion.div
          className="flex items-center gap-2 text-xs font-mono px-3 py-1.5 rounded-full border border-real/30 bg-real/10"
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
        >
          <motion.div
            className="w-1.5 h-1.5 rounded-full bg-real-light"
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
          <span className="text-real-light">API online</span>
          <Activity size={12} className="text-slate-500" />
        </motion.div>
      </div>
    </header>
  );
}
