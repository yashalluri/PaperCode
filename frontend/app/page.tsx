"use client";

import { useState, useRef, useEffect, useCallback } from "react";

// ─── Types ──────────────────────────────────────────────────────────
interface ComparisonMetric {
  claimed: number;
  reproduced: number | null;
  absolute_difference?: number;
  difference_description?: string;
  within_tolerance?: boolean;
  status?: string;
}

interface Results {
  metrics?: Record<string, number>;
  comparison?: {
    comparisons: Record<string, ComparisonMetric>;
    summary: {
      total_metrics: number;
      within_tolerance: number;
      outside_tolerance: number;
      verdict: string;
      verdict_emoji: string;
    };
  };
  report?: string;
  figures?: string[];
}

// ─── Phase config ───────────────────────────────────────────────────
const PHASES: Record<string, { label: string; icon: string }> = {
  starting: { label: "Starting", icon: "⏳" },
  workspace: { label: "Creating workspace", icon: "📁" },
  fetching: { label: "Fetching paper", icon: "📥" },
  parsing: { label: "Parsing PDF", icon: "📄" },
  analyzing: { label: "Analyzing paper", icon: "🔬" },
  generating: { label: "Generating code", icon: "💻" },
  installing: { label: "Installing deps", icon: "📦" },
  training: { label: "Training models", icon: "🧠" },
  evaluating: { label: "Evaluating", icon: "📊" },
  comparing: { label: "Comparing results", icon: "⚖️" },
  reporting: { label: "Generating report", icon: "📝" },
  done: { label: "Complete", icon: "✅" },
};

// ─── Main Page ──────────────────────────────────────────────────────
export default function Home() {
  const [url, setUrl] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [phase, setPhase] = useState<string>("starting");
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [title, setTitle] = useState<string | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [error, setError] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // SSE connection
  const connectSSE = useCallback((id: string) => {
    const es = new EventSource(`/api/jobs/${id}/stream`);

    es.addEventListener("update", (e) => {
      const data = JSON.parse(e.data);
      setStatus(data.status);
      setPhase(data.phase);
      setProgress(data.progress);
      if (data.title) setTitle(data.title);
      if (data.logs?.length) {
        setLogs((prev) => [...prev, ...data.logs]);
      }
    });

    es.addEventListener("complete", (e) => {
      const data = JSON.parse(e.data);
      setStatus(data.status);
      if (data.results) setResults(data.results);
      if (data.error) setError(data.error);
      setProgress(100);
      es.close();
    });

    es.onerror = () => {
      es.close();
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;

    // Reset state
    setLogs([]);
    setResults(null);
    setError(null);
    setTitle(null);
    setStatus("running");
    setPhase("starting");
    setProgress(0);

    try {
      const res = await fetch("/api/replicate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_url: url }),
      });
      const data = await res.json();
      setJobId(data.job_id);
      connectSSE(data.job_id);
    } catch {
      setError("Failed to connect to API. Is the backend running?");
      setStatus("error");
    }
  };

  const isRunning = status === "running";
  const isDone = status === "completed";
  const isError = status === "error";

  return (
    <main className="max-w-5xl mx-auto px-6 py-12">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold mb-3 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
          Paper Replicator
        </h1>
        <p className="text-gray-400 text-lg">
          Paste an arXiv URL. Get a reproducibility verdict.
        </p>
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="mb-10">
        <div className="flex gap-3">
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://arxiv.org/abs/2011.14439"
            disabled={isRunning}
            className="flex-1 px-5 py-4 bg-[#12121a] border border-gray-700/50 rounded-xl text-white text-lg placeholder-gray-500 focus:outline-none focus:border-purple-500/50 focus:ring-1 focus:ring-purple-500/30 transition-all disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isRunning || !url.trim()}
            className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-500 hover:to-purple-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed text-lg"
          >
            {isRunning ? "Running..." : "Replicate"}
          </button>
        </div>
      </form>

      {/* Pipeline status */}
      {status !== "idle" && (
        <div className="space-y-6">
          {/* Paper title */}
          {title && (
            <div className="bg-[#12121a] rounded-xl p-5 border border-gray-800/50">
              <p className="text-sm text-gray-400 mb-1">Paper</p>
              <p className="text-xl font-medium">{title}</p>
            </div>
          )}

          {/* Progress bar */}
          <div className="bg-[#12121a] rounded-xl p-5 border border-gray-800/50">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="text-lg">
                  {PHASES[phase]?.icon || "⏳"}
                </span>
                <span className="font-medium">
                  {PHASES[phase]?.label || phase}
                </span>
              </div>
              <span className="text-gray-400 text-sm">{progress}%</span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-500 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>

            {/* Phase indicators */}
            <div className="flex flex-wrap gap-2 mt-4">
              {Object.entries(PHASES).map(([key, val]) => {
                const phaseKeys = Object.keys(PHASES);
                const currentIdx = phaseKeys.indexOf(phase);
                const thisIdx = phaseKeys.indexOf(key);
                const isPast = thisIdx < currentIdx;
                const isCurrent = key === phase;

                return (
                  <span
                    key={key}
                    className={`text-xs px-2.5 py-1 rounded-full transition-all ${
                      isCurrent
                        ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                        : isPast
                        ? "bg-green-500/10 text-green-400/70"
                        : "bg-gray-800/50 text-gray-600"
                    }`}
                  >
                    {isPast ? "✓" : val.icon} {val.label}
                  </span>
                );
              })}
            </div>
          </div>

          {/* Live logs */}
          <div className="bg-[#12121a] rounded-xl border border-gray-800/50 overflow-hidden">
            <div className="px-5 py-3 border-b border-gray-800/50 flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isRunning ? "bg-green-400 animate-pulse" : isDone ? "bg-blue-400" : "bg-red-400"}`} />
              <span className="text-sm font-medium text-gray-300">Pipeline Output</span>
            </div>
            <div className="p-5 max-h-80 overflow-y-auto font-mono text-sm space-y-0.5">
              {logs.map((log, i) => (
                <div
                  key={i}
                  className={`${
                    log.startsWith("Error")
                      ? "text-red-400"
                      : log.includes("===")
                      ? "text-purple-400 font-bold"
                      : log.includes("final:")
                      ? "text-green-400"
                      : log.includes("✅")
                      ? "text-green-400"
                      : log.includes("❌")
                      ? "text-red-400"
                      : "text-gray-400"
                  }`}
                >
                  {log}
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>

          {/* Error */}
          {isError && error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-5">
              <p className="text-red-400 font-medium">Error</p>
              <p className="text-red-300 text-sm mt-1">{error}</p>
            </div>
          )}

          {/* Results */}
          {isDone && results && <ResultsPanel results={results} jobId={jobId!} />}
        </div>
      )}
    </main>
  );
}

// ─── Results Panel ──────────────────────────────────────────────────
function ResultsPanel({ results, jobId }: { results: Results; jobId: string }) {
  const comparison = results.comparison;
  const verdict = comparison?.summary;

  return (
    <div className="space-y-6">
      {/* Verdict Banner */}
      {verdict && (
        <div
          className={`rounded-xl p-6 border ${
            verdict.verdict === "REPRODUCED"
              ? "bg-green-500/10 border-green-500/30"
              : verdict.verdict === "PARTIALLY_REPRODUCED"
              ? "bg-yellow-500/10 border-yellow-500/30"
              : verdict.verdict === "NOT_REPRODUCED"
              ? "bg-red-500/10 border-red-500/30"
              : "bg-gray-500/10 border-gray-500/30"
          }`}
        >
          <div className="text-center">
            <p className="text-4xl mb-2">{verdict.verdict_emoji}</p>
            <h2 className="text-2xl font-bold">
              {verdict.verdict.replace(/_/g, " ")}
            </h2>
            <p className="text-gray-400 mt-1">
              {verdict.within_tolerance} of {verdict.total_metrics} metrics within tolerance
            </p>
          </div>
        </div>
      )}

      {/* Metrics Comparison Table */}
      {comparison && (
        <div className="bg-[#12121a] rounded-xl border border-gray-800/50 overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-800/50">
            <span className="font-medium text-gray-300">Results Comparison</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 text-left">
                  <th className="px-5 py-3 font-medium">Metric</th>
                  <th className="px-5 py-3 font-medium">Paper</th>
                  <th className="px-5 py-3 font-medium">Reproduced</th>
                  <th className="px-5 py-3 font-medium">Difference</th>
                  <th className="px-5 py-3 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(comparison.comparisons).map(([name, m]) => (
                  <tr key={name} className="border-t border-gray-800/30">
                    <td className="px-5 py-3 font-mono text-gray-300">
                      {name.replace(/_/g, " ")}
                    </td>
                    <td className="px-5 py-3">{m.claimed?.toFixed(1)}%</td>
                    <td className="px-5 py-3">
                      {m.reproduced !== null ? `${m.reproduced.toFixed(1)}%` : "N/A"}
                    </td>
                    <td className="px-5 py-3 text-gray-400 text-xs">
                      {m.difference_description || "—"}
                    </td>
                    <td className="px-5 py-3 text-lg">
                      {m.within_tolerance ? "✅" : m.status === "MISSING" ? "⚠️" : "❌"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Figures */}
      {results.figures && results.figures.length > 0 && (
        <div className="bg-[#12121a] rounded-xl border border-gray-800/50 overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-800/50">
            <span className="font-medium text-gray-300">Reproduced Figures</span>
          </div>
          <div className="p-5 grid gap-4">
            {results.figures.map((fig) => (
              <div key={fig} className="bg-white rounded-lg p-2">
                <img
                  src={`/api/jobs/${jobId}/figures/${fig}`}
                  alt={fig}
                  className="w-full rounded"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Report */}
      {results.report && (
        <div className="bg-[#12121a] rounded-xl border border-gray-800/50 overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-800/50">
            <span className="font-medium text-gray-300">Full Report</span>
          </div>
          <div className="p-5 max-h-96 overflow-y-auto">
            <pre className="text-sm text-gray-400 whitespace-pre-wrap font-mono">
              {results.report}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
