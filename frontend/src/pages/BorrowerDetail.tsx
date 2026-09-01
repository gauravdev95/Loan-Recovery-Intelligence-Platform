import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import type {
  BorrowerDetail as Detail,
  ExplanationFactor,
  ExplanationResponse,
} from "../api/types";
import { PriorityBadge, Probability, RiskBadge } from "../components/badges";
import { ErrorState, Spinner } from "../components/states";

export default function BorrowerDetail() {
  const { id = "" } = useParams();
  const [detail, setDetail] = useState<Detail | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const d = await api.getBorrower(id);
        if (!active) return;
        setDetail(d);
        // Build an explain request from stored features.
        if (d.features) {
          const payload = {
            borrower_id: d.borrower_id,
            RevolvingUtilizationOfUnsecuredLines:
              d.features.revolving_utilization ?? 0,
            age: d.features.age ?? 40,
            NumberOfTime30_59DaysPastDueNotWorse:
              d.features.num_30_59_past_due ?? 0,
            DebtRatio: d.features.debt_ratio ?? 0,
            MonthlyIncome: d.features.monthly_income,
            NumberOfOpenCreditLinesAndLoans:
              d.features.open_credit_lines ?? 0,
            NumberOfTimes90DaysLate: d.features.num_90_late ?? 0,
            NumberRealEstateLoansOrLines: d.features.real_estate_loans ?? 0,
            NumberOfTime60_89DaysPastDueNotWorse:
              d.features.num_60_89_past_due ?? 0,
            NumberOfDependents: d.features.dependents,
          };
          const ex = await api.explain(payload);
          if (active) setExplanation(ex);
        }
      } catch (e) {
        if (active) {
          setError(e instanceof Error ? e.message : "Failed to load borrower");
        }
      } finally {
        if (active) setLoading(false);
      }
    }
    void load();
    return () => {
      active = false;
    };
  }, [id]);

  if (loading) return <Spinner label="Loading borrower…" />;
  if (error) return <ErrorState message={error} />;
  if (!detail) return <ErrorState message="Borrower not found." />;

  const label = (k: string) =>
    ({
      revolving_utilization: "Revolving utilization",
      age: "Age",
      num_30_59_past_due: "30-59 days past due",
      debt_ratio: "Debt ratio",
      monthly_income: "Monthly income",
      open_credit_lines: "Open credit lines",
      num_90_late: "90+ days late",
      real_estate_loans: "Real estate loans",
      num_60_89_past_due: "60-89 days past due",
      dependents: "Dependents",
    }[k] ?? k);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link
          to="/borrowers"
          className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50"
        >
          ← Back
        </Link>
        <h2 className="text-xl font-semibold text-slate-900">
          {detail.borrower_id}
        </h2>
        <RiskBadge level={detail.risk_level} />
        <PriorityBadge priority={detail.recovery_priority} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-700">
              Default probability
            </h3>
            <Probability value={detail.default_probability} />
          </div>
          <p className="mt-4 text-5xl font-bold text-brand-700">
            {(detail.default_probability ?? 0) * 100 > 0
              ? ((detail.default_probability ?? 0) * 100).toFixed(1)
              : "—"}
            {(detail.default_probability ?? 0) * 100 > 0 ? "%" : ""}
          </p>
          <p className="mt-2 text-sm text-slate-500">
            {detail.recommended_action ?? "No recommendation recorded."}
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h3 className="mb-3 text-sm font-semibold text-slate-700">
            Key risk factors (SHAP)
          </h3>
          {explanation ? (
            <ul className="space-y-2">
              {explanation.top_factors.map((f: ExplanationFactor) => (
                <li
                  key={f.feature}
                  className="flex items-center justify-between rounded-md bg-slate-50 px-3 py-2 text-sm"
                >
                  <span className="font-medium">{f.feature}</span>
                  <span
                    className={
                      f.direction === "increases_risk"
                        ? "text-red-600"
                        : "text-emerald-600"
                    }
                  >
                    {f.direction === "increases_risk" ? "▲" : "▼"}{" "}
                    {f.shap_value?.toFixed(3) ?? "—"}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-slate-400">Explanation unavailable.</p>
          )}
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <h3 className="mb-3 text-sm font-semibold text-slate-700">
          Borrower profile
        </h3>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
          {Object.entries(detail.features ?? {}).map(([k, v]) => (
            <div
              key={k}
              className="rounded-md border border-slate-100 bg-slate-50 px-3 py-2"
            >
              <p className="text-xs text-slate-500">{label(k)}</p>
              <p className="font-mono text-sm font-medium">
                {v === null || v === undefined ? "—" : v.toLocaleString()}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
