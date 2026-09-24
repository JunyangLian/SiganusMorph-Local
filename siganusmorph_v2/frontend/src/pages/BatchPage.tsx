import { ChangeEvent, useMemo, useState } from "react";
import { calibrate, exportSession, measure, rememberExport, uploadImage } from "../api/client";

type BatchStatus = "waiting" | "uploading" | "calibrating" | "measuring" | "exporting" | "exported" | "calibration_failed" | "needs_review";

type Row = {
  file: File;
  name: string;
  status: BatchStatus;
  session_id?: string;
  saved_to?: string;
  message?: string;
};

const statusLabel: Record<BatchStatus, string> = {
  waiting: "等待处理",
  uploading: "上传中",
  calibrating: "校准中",
  measuring: "测量中",
  exporting: "导出中",
  exported: "已导出",
  calibration_failed: "校准失败",
  needs_review: "需要复核",
};

function statusClass(status: BatchStatus) {
  if (status === "exported") return "success";
  if (status === "calibration_failed" || status === "needs_review") return "error";
  return "";
}

export function BatchPage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [running, setRunning] = useState(false);
  const summary = useMemo(() => {
    return rows.reduce(
      (acc, row) => {
        acc[row.status] = (acc[row.status] ?? 0) + 1;
        return acc;
      },
      {} as Record<string, number>,
    );
  }, [rows]);

  function patchRow(index: number, patch: Partial<Row>) {
    setRows((prev) => prev.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  }

  async function runBatch() {
    setRunning(true);
    for (let i = 0; i < rows.length; i += 1) {
      const row = rows[i];
      try {
        patchRow(i, { status: "uploading", message: "" });
        const uploaded = await uploadImage(row.file, "", "");
        patchRow(i, { session_id: uploaded.session_id, status: "calibrating" });
        const cal = await calibrate(uploaded.session_id);
        if (cal.status !== "success") {
          patchRow(i, { status: "calibration_failed", message: cal.message || "校准失败，已阻止测量。" });
          continue;
        }
        patchRow(i, { status: "measuring" });
        await measure(uploaded.session_id);
        patchRow(i, { status: "exporting" });
        const exported = await exportSession(uploaded.session_id);
        rememberExport({
          ...exported,
          image_name: uploaded.image_name,
          specimen_id: "",
          exported_at: new Date().toISOString(),
        });
        patchRow(i, { status: "exported", saved_to: exported.saved_to });
      } catch (exc) {
        patchRow(i, {
          status: "needs_review",
          message: exc instanceof Error ? exc.message : String(exc),
        });
      }
    }
    setRunning(false);
  }

  return (
    <>
      <section className="hero">
        <div className="eyebrow">Batch Measurement</div>
        <h1>批量测量</h1>
        <p>批量上传图片，逐张完成上传、校准、自动测量和导出。校准失败的图片会被阻止继续测量，并标记为需要复核。</p>
      </section>
      <div className="grid two">
        <div className="card">
          <h2 className="section-title">上传队列</h2>
          <div className="field">
            <label>上传多张图片</label>
            <input
              type="file"
              accept="image/*,.heic"
              multiple
              onChange={(event: ChangeEvent<HTMLInputElement>) => {
                const files = Array.from(event.target.files ?? []);
                setRows(files.map((file) => ({ file, name: file.name, status: "waiting" })));
              }}
            />
          </div>
          <div className="button-row">
            <button className="btn primary" disabled={!rows.length || running} onClick={runBatch}>
              开始批量测量
            </button>
            <span className="muted">输出统一写入 results/v2_user_outputs/。</span>
          </div>
        </div>
        <div className="card">
          <h2 className="section-title">处理摘要</h2>
          <div className="metrics">
            {Object.entries(summary).map(([status, count]) => (
              <div className="metric" key={status}>
                <div className="metric-label">{statusLabel[status as BatchStatus] ?? status}</div>
                <div className="metric-value">{count}</div>
              </div>
            ))}
            {!rows.length && <p className="muted">尚未选择图片。</p>}
          </div>
        </div>
      </div>
      <div className="card" style={{ marginTop: 22 }}>
        <h2 className="section-title">图片处理状态</h2>
        <table className="table">
          <thead>
            <tr>
              <th>image_name</th>
              <th>status</th>
              <th>output</th>
              <th>notes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name}>
                <td>{row.name}</td>
                <td>
                  <span className={`status ${statusClass(row.status)}`}>{statusLabel[row.status]}</span>
                </td>
                <td>{row.saved_to ?? "-"}</td>
                <td>{row.message ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
