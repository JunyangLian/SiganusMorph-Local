import { useEffect, useState } from "react";
import { clearExportHistory, ExportHistoryEntry, listExports, loadExportHistory } from "../api/client";

export function ExportPage() {
  const [history, setHistory] = useState<ExportHistoryEntry[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    listExports()
      .then((response) => setHistory(response.exports))
      .catch((exc) => {
        setError(exc instanceof Error ? exc.message : String(exc));
        setHistory(loadExportHistory());
      });
  }, []);

  function clearHistory() {
    clearExportHistory();
    setHistory([]);
  }

  return (
    <>
      <section className="hero">
        <div className="eyebrow">Export Center</div>
        <h1>结果导出</h1>
        <p>集中查看本机用户端已导出的 CSV、Excel、JSON 和 preview 记录。V2.0 文件统一写入 results/v2_user_outputs/。</p>
      </section>
      <div className="card">
        <div className="button-row" style={{ justifyContent: "space-between" }}>
          <div>
            <h2 className="section-title">导出历史</h2>
            <p className="muted">这里显示 V2 用户端导出结果。若后端暂不可用，将回退到当前浏览器记录；不读取或修改历史 V1.0 数据。</p>
          </div>
          <button className="btn" onClick={clearHistory} disabled={!history.length}>
            清空列表
          </button>
        </div>
        {error && <div className="error-box" style={{ marginBottom: 16 }}>{error}</div>}
        <table className="table">
          <thead>
            <tr>
              <th>image_name</th>
              <th>specimen_id</th>
              <th>weight_g</th>
              <th>exported_at</th>
              <th>saved_to</th>
              <th>files</th>
            </tr>
          </thead>
          <tbody>
            {history.map((entry) => (
              <tr key={entry.session_id}>
                <td>{entry.image_name ?? "-"}</td>
                <td>{entry.specimen_id ?? ""}</td>
                <td>{entry.weight_g ?? ""}</td>
                <td>{typeof entry.exported_at === "number" ? new Date(entry.exported_at * 1000).toLocaleString() : new Date(entry.exported_at).toLocaleString()}</td>
                <td>{entry.saved_to}</td>
                <td>{Object.keys(entry.files).join(", ")}</td>
              </tr>
            ))}
            {!history.length && (
              <tr>
                <td colSpan={6}>暂无导出记录。请先在单鱼测量或批量测量页面完成导出。</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}
