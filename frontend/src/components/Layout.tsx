import { NavLink, Outlet } from "react-router-dom";

const NAV = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/borrowers", label: "Borrowers" },
  { to: "/recovery", label: "Recovery Queue" },
  { to: "/performance", label: "Model Performance" },
];

export default function Layout() {
  return (
    <div className="min-h-full flex flex-col">
      <header className="bg-brand-900 text-white shadow-md">
        <div className="mx-auto max-w-7xl px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-lg bg-white/15 grid place-items-center font-bold">
              LR
            </div>
            <div>
              <h1 className="text-lg font-semibold leading-tight">
                Loan Recovery Intelligence
              </h1>
              <p className="text-xs text-brand-100">
                ML default-risk &amp; recovery decision support
              </p>
            </div>
          </div>
          <nav className="hidden md:flex gap-1">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-white/20 text-white"
                      : "text-brand-100 hover:bg-white/10 hover:text-white"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
        {/* mobile nav */}
        <nav className="md:hidden flex gap-1 px-4 pb-3 overflow-x-auto">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `whitespace-nowrap rounded-md px-3 py-2 text-sm font-medium ${
                  isActive ? "bg-white/20 text-white" : "text-brand-100"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="flex-1 mx-auto w-full max-w-7xl px-4 py-6">
        <Outlet />
      </main>
      <footer className="border-t border-slate-200 bg-white py-4 text-center text-xs text-slate-400">
        Loan Recovery Intelligence Platform — educational / prototype decision-support,
        not an automated lending or collections system.
      </footer>
    </div>
  );
}
