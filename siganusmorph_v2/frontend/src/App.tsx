import { useMemo, useState } from "react";
import { BatchPage } from "./pages/BatchPage";
import { ExportPage } from "./pages/ExportPage";
import { HomePage } from "./pages/HomePage";
import { SettingsPage } from "./pages/SettingsPage";
import { SingleFishPage } from "./pages/SingleFishPage";

type PageKey = "home" | "single" | "batch" | "export" | "settings";

const NAV: Array<{ key: PageKey; label: string }> = [
  { key: "home", label: "首页" },
  { key: "single", label: "单鱼测量" },
  { key: "batch", label: "批量测量" },
  { key: "export", label: "结果导出" },
  { key: "settings", label: "设置" },
];

export function App() {
  const [page, setPage] = useState<PageKey>("home");
  const content = useMemo(() => {
    if (page === "single") return <SingleFishPage />;
    if (page === "batch") return <BatchPage />;
    if (page === "export") return <ExportPage />;
    if (page === "settings") return <SettingsPage />;
    return <HomePage onNavigate={setPage} />;
  }, [page]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-title">SiganusMorph V2.0</div>
        <div className="brand-subtitle">蓝子鱼形态测量系统</div>
        <nav className="nav">
          {NAV.map((item) => (
            <button key={item.key} className={page === item.key ? "active" : ""} onClick={() => setPage(item.key)}>
              {item.label}
            </button>
          ))}
        </nav>
      </aside>
      <main className="main">{content}</main>
    </div>
  );
}
