import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { BorrowerRow } from "../api/types";
import { PriorityBadge, Probability, RiskBadge } from "../components/badges";
import { EmptyState, ErrorState, Spinner } from "../components/states";

const PAGE_SIZE = 25;

export default function Borrowers() {
  const [items, setItems] = useState<BorrowerRow[]>([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [risk, setRisk] = useState("");
  const [priority, setPriority] = useState("");
  const [sortBy, setSortBy] = useState("created_at");
  const [order, setOrder] = useState("desc");
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.listBorrowers({
        search: search || undefined,
        risk_level: risk || undefined,
        priority: priority || undefined,
        sort_by: sortBy,
        order,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      });
      setItems(res.items);
      setTotal(res.total);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load borrowers");
    } finally {
      setLoading(false);
    }
  }, [search, risk, priority, sortBy, order, page]);

  useEffect(() => {
    void load();
  }, [load]);

  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold text-slate-900">Borrowers</h2>
          <p className="text-sm text-slate-500">{total.toLocaleString()} borrowers scored</p>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
        <input
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(0);
          }}
          placeholder="Search borrower ID…"
          className="rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
        />
        <select
          value={risk}
          onChange={(e) => {
            setRisk(e.target.value);
            setPage(0);
          }}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm"
          aria-label="Filter by risk"
        >
          <option value="">All risk levels</option>
          <option>LOW</option>
          <option>MEDIUM</option>
          <option>HIGH</option>
          <option>CRITICAL</option>
        </select>
        <select
          value={priority}
          onChange={(e) => {
            setPriority(e.target.value);
            setPage(0);
          }}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm"
          aria-label="Filter by priority"
        >
          <option value="">All priorities</option>
          <option>NORMAL</option>
          <option>FOLLOW_UP</option>
          <option>HIGH_PRIORITY</option>
          <option>CRITICAL_PRIORITY</option>
        </select>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm"
          aria-label="Sort by"
        >
          <option value="created_at">Created</option>
          <option value="probability">Probability</option>
          <option value="risk_level">Risk level</option>
          <option value="priority">Priority</option>
        </select>
        <button
          onClick={() => setOrder((o) => (o === "desc" ? "asc" : "desc"))}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm hover:bg-slate-50"
        >
          {order === "desc" ? "↓ Desc" : "↑ Asc"}
        </button>
      </div>

      {loading ? (
        <Spinner label="Loading borrowers…" />
      ) : error ? (
        <ErrorState message={error} onRetry={load} />
      ) : items.length === 0 ? (
        <EmptyState message="No borrowers match the current filters." />
      ) : (
        <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-3">Borrower</th>
                <th className="px-4 py-3">Default prob</th>
                <th className="px-4 py-3">Risk</th>
                <th className="px-4 py-3">Recovery priority</th>
                <th className="px-4 py-3">Model</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {items.map((b) => (
                <tr key={b.borrower_id} className="hover:bg-slate-50">
                  <td className="px-4 py-3">
                    <Link
                      to={`/borrowers/${b.borrower_id}`}
                      className="font-medium text-brand-600 hover:underline"
                    >
                      {b.borrower_id}
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <Probability value={b.default_probability} />
                  </td>
                  <td className="px-4 py-3">
                    <RiskBadge level={b.risk_level} />
                  </td>
                  <td className="px-4 py-3">
                    <PriorityBadge priority={b.recovery_priority} />
                  </td>
                  <td className="px-4 py-3 text-xs text-slate-500">
                    {b.model_version ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex items-center justify-between text-sm text-slate-500">
        <span>
          Page {page + 1} of {pages}
        </span>
        <div className="flex gap-2">
          <button
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
            className="rounded-md border border-slate-300 px-3 py-1.5 disabled:opacity-40 hover:bg-slate-50"
          >
            Prev
          </button>
          <button
            disabled={page + 1 >= pages}
            onClick={() => setPage((p) => p + 1)}
            className="rounded-md border border-slate-300 px-3 py-1.5 disabled:opacity-40 hover:bg-slate-50"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
