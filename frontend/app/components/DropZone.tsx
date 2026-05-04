"use client";

import { useCallback, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, ImageIcon, X } from "lucide-react";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function DropZone({ onFile, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = (file: File) => {
    if (!file.type.startsWith("image/")) return;
    setPreview(URL.createObjectURL(file));
    setFilename(file.name);
    onFile(file);
  };

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file) accept(file);
    },
    [disabled] // eslint-disable-line react-hooks/exhaustive-deps
  );

  const clear = (e: React.MouseEvent) => {
    e.stopPropagation();
    setPreview(null);
    setFilename(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div
      className={`relative w-full rounded-2xl border-2 border-dashed transition-colors cursor-pointer select-none
        ${dragging ? "border-brand-500 bg-brand-500/10" : "border-surface-border bg-surface-card hover:border-brand-500/50"}
        ${disabled ? "opacity-50 cursor-not-allowed" : ""}
      `}
      style={{ minHeight: "260px" }}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) accept(file);
        }}
      />

      <AnimatePresence mode="wait">
        {preview ? (
          <motion.div
            key="preview"
            className="absolute inset-0 rounded-2xl overflow-hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={preview}
              alt="Preview"
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-surface/50" />
            <div className="absolute bottom-3 left-3 right-3 flex items-center gap-2">
              <span className="text-xs text-slate-300 font-mono truncate flex-1">
                {filename}
              </span>
              <button
                onClick={clear}
                className="p-1 rounded-full bg-surface-card/80 hover:bg-fake/20 transition-colors"
              >
                <X size={14} className="text-slate-400" />
              </button>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="empty"
            className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="p-4 rounded-full border border-surface-border bg-surface-muted"
              animate={dragging ? { scale: 1.15, borderColor: "#4f6ef7" } : { scale: 1 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              {dragging ? (
                <ImageIcon size={28} className="text-brand-500" />
              ) : (
                <Upload size={28} className="text-slate-500" />
              )}
            </motion.div>
            <div className="text-center">
              <p className="text-slate-300 font-medium">
                {dragging ? "Drop to analyse" : "Drop an image here"}
              </p>
              <p className="text-slate-500 text-sm mt-1">
                or click to browse &mdash; JPG, PNG, WEBP
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
