type PageKey = "home" | "single" | "batch" | "export" | "settings";

export function HomePage({ onNavigate }: { onNavigate: (page: PageKey) => void }) {
  return (
    <>
      <section className="hero">
        <div className="eyebrow">SiganusMorph Local V2.0</div>
        <h1>蓝子鱼形态参数智能测量系统</h1>
        <p>
          面向标准化单鱼侧位图像的桌面测量应用。V2.0 将 Python 测量核心、FastAPI 后端、React 前端和
          Tauri 桌面壳分层组织，让普通用户使用更流畅，管理者仍可保留完整复核工具。
        </p>
        <div className="button-row">
          <button className="btn primary" onClick={() => onNavigate("single")}>开始单鱼测量</button>
          <button className="btn secondary" onClick={() => onNavigate("batch")}>批量测量</button>
          <button className="btn secondary" onClick={() => onNavigate("export")}>结果导出</button>
        </div>
      </section>
      <div className="grid three">
        <div className="card">
          <h2 className="section-title">自动校准</h2>
          <p className="muted">上传原始照片后，后端检测校准标记并生成 warped image。校准失败时不会继续测量。</p>
        </div>
        <div className="card">
          <h2 className="section-title">测量工作台</h2>
          <p className="muted">在 warped image 上拖拽正式测量点，支持 hover label、crosshair、放大镜和键盘微调。</p>
        </div>
        <div className="card">
          <h2 className="section-title">本地导出</h2>
          <p className="muted">导出 CSV、Excel、JSON 和标注预览图，统一写入 V2 用户输出目录，不覆盖 V1 历史结果。</p>
        </div>
      </div>
    </>
  );
}
