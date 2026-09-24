import { useEffect, useState } from "react";
import { API_BASE, getHealth, getSettings, HealthResponse, SettingsResponse } from "../api/client";

export function SettingsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getHealth().then(setHealth).catch((exc) => setError(exc instanceof Error ? exc.message : String(exc)));
    getSettings().then(setSettings).catch((exc) => setError(exc instanceof Error ? exc.message : String(exc)));
  }, []);

  return (
    <>
      <section className="hero">
        <div className="eyebrow">Settings</div>
        <h1>设置</h1>
        <p>查看后端连接、版本信息、输出目录和未来桌面打包说明。</p>
      </section>
      <div className="card">
        <h2 className="section-title">后端连接</h2>
        <p className="muted">API Base: {API_BASE}</p>
        {health && <p className="success-box">后端状态：{health.status} | version {health.version}</p>}
        {error && <p className="error-box">{error}</p>}
      </div>
      <div className="grid two" style={{ marginTop: 22 }}>
        <div className="card">
          <h2 className="section-title">本地测量引擎</h2>
          <p className="muted">
            V2.0 复用现有 Python measurement core：{settings?.core_engine ?? "siganusmorph/"}。模型权重和几何规则由后端统一调用，用户端不直接暴露调试参数。
          </p>
          <table className="table">
            <thead>
              <tr>
                <th>模型</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              {(settings?.model_paths ?? []).map((model) => (
                <tr key={model.name}>
                  <td>{model.name}</td>
                  <td><span className={`status ${model.exists ? "success" : "warning"}`}>{model.exists ? "可用" : "未找到"}</span></td>
                </tr>
              ))}
              {!settings && (
                <tr>
                  <td colSpan={2}>正在读取后端设置...</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="card">
          <h2 className="section-title">输出目录</h2>
          <p className="muted">
            用户端导出统一写入 {settings?.user_output_root ?? "results/v2_user_outputs/"}。管理者端后续任务应写入 {settings?.admin_output_root ?? "results/v2_admin_outputs/"}。
          </p>
          {settings?.notes.map((note) => <p className="muted" key={note}>{note}</p>)}
        </div>
      </div>
    </>
  );
}
