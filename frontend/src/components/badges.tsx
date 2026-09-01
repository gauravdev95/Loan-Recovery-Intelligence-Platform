import type { RecoveryPriority, RiskLevel } from "../api/types";

const RISK_STYLES: Record<RiskLevel, string> = {
  LOW: "bg-emerald-100 text-emerald-800",
  MEDIUM: "bg-amber-100 text-amber-800",
  HIGH: "bg-orange-100 text-orange-800",
  CRITICAL: "bg-red-100 text-red-800",
};

const PRIORITY_STYLES: Record<RecoveryPriority, string> = {
  NORMAL: "bg-slate-200 text-slate-700",
  FOLLOW_UP: "bg-amber-100 text-amber-800",
  HIGH_PRIORITY: "bg-orange-100 text-orange-800",
  CRITICAL_PRIORITY: "bg-red-200 text-red-900",
};

export function RiskBadge({ level }: { level: RiskLevel | null }) {
  if (!level) return <span className="text-xs text-slate-400">—</span>;
  const style = RISK_STYLES[level] ?? RISK_STYLES.LOW;
  return (
    <span
      className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold ${style}`}
    >
      {level}
    </span>
  );
}

export function PriorityBadge({ priority }: { priority: RecoveryPriority | null }) {
  if (!priority)
    return <span className="text-xs text-slate-400">—</span>;
  const style = PRIORITY_STYLES[priority] ?? PRIORITY_STYLES.NORMAL;
  return (
    <span
      className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold ${style}`}
    >
      {priority.replace("_", " ")}
    </span>
  );
}

export function Probability({ value }: { value: number | null }) {
  if (value === null || value === undefined)
    return <span className="text-slate-400">—</span>;
  return <span className="font-mono font-medium">{(value * 100).toFixed(1)}%</span>;
}
