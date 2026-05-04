import Header from "./components/Header";
import DeepfakeDetector from "./components/DeepfakeDetector";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-surface grid-bg">
      <Header />

      <main className="flex-1 px-6 py-10 max-w-5xl mx-auto w-full">
        {/* Hero text */}
        <div className="mb-10 text-center">
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight mb-3">
            Detect{" "}
            <span className="text-transparent bg-clip-text bg-linear-to-r from-brand-500 to-purple-400">
              deepfakes
            </span>{" "}
            instantly
          </h1>
          <p className="text-slate-400 text-lg max-w-xl mx-auto">
            Powered by{" "}
            <span className="text-slate-300 font-medium">EfficientNetB4</span>,{" "}
            <span className="text-slate-300 font-medium">Attention Mechanisms</span> &amp;{" "}
            <span className="text-slate-300 font-medium">Vision Transformers</span>
          </p>

          {/* Accuracy badges */}
          <div className="flex flex-wrap justify-center gap-3 mt-5">
            {[
              { label: "FaceForensics++", value: "92.4%" },
              { label: "DFDC", value: "89.7%" },
              { label: "COCOFake", value: "87.4%" },
            ].map(({ label, value }) => (
              <div
                key={label}
                className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs border border-surface-border bg-surface-card"
              >
                <span className="text-slate-400">{label}</span>
                <span className="font-mono font-semibold text-brand-500">{value}</span>
              </div>
            ))}
          </div>
        </div>

        <DeepfakeDetector />
      </main>

      <footer className="text-center text-xs text-slate-600 py-6 border-t border-surface-border">
        DeepScan &mdash; Final Year Project &bull; EfficientNetB4 + Attention + ViT
      </footer>
    </div>
  );
}
