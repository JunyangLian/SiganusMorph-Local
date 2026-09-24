import { ChangeEvent, useEffect, useState } from "react";
import {
  assetUrl,
  calibrate,
  CalibrateResponse,
  exportSession,
  ExportResponse,
  getSession,
  measure,
  MeasureResponse,
  rememberExport,
  updatePoints,
  uploadImage,
  UploadResponse,
} from "../api/client";
import { MeasurementCanvas, PointMap } from "../components/MeasurementCanvas";
import { MetricCards } from "../components/MetricCards";

export function SingleFishPage() {
  const [file, setFile] = useState<File | null>(null);
  const [specimenId, setSpecimenId] = useState("");
  const [weightG, setWeightG] = useState("");
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [calibration, setCalibration] = useState<CalibrateResponse | null>(null);
  const [measurement, setMeasurement] = useState<MeasureResponse | null>(null);
  const [exportInfo, setExportInfo] = useState<ExportResponse | null>(null);
  const [restoreSessionId, setRestoreSessionId] = useState("");
  const [status, setStatus] = useState("未上传");
  const [error, setError] = useState("");

  useEffect(() => {
    if (measurement) setStatus("测量完成");
    else if (calibration?.status === "failed") setStatus("校准失败");
    else if (upload) setStatus("等待测量");
    else if (!file) setStatus("未上传");
  }, [file, upload, calibration, measurement]);

  async function runPipeline() {
    if (!file) return;
    setError("");
    setExportInfo(null);
    setStatus("上传中");
    try {
      const uploaded = await uploadImage(file, specimenId, weightG);
      setUpload(uploaded);
      setStatus("校准中");
      const cal = await calibrate(uploaded.session_id);
      setCalibration(cal);
      if (cal.status !== "success") {
        setError(cal.message || "校准失败，无法继续自动测量。");
        return;
      }
      setStatus("测量中");
      const measured = await measure(uploaded.session_id);
      setMeasurement(measured);
      setStatus("测量完成");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
      setStatus("需要复核");
    }
  }

  async function applyPoints(points: PointMap) {
    if (!upload) return;
    const updated = await updatePoints(upload.session_id, points);
    setMeasurement(updated);
  }

  async function doExport() {
    if (!upload) return;
    const exported = await exportSession(upload.session_id);
    setExportInfo(exported);
    rememberExport({
      ...exported,
      image_name: upload.image_name,
      specimen_id: specimenId,
      weight_g: weightG,
      exported_at: new Date().toISOString(),
    });
  }

  async function restoreSession() {
    const id = restoreSessionId.trim();
    if (!id) return;
    setError("");
    setExportInfo(null);
    try {
      const restored = await getSession(id);
      setUpload({
        version: restored.version,
        session_id: restored.session_id,
        image_name: restored.image_name,
        raw_image_url: restored.raw_image_url ?? "",
        status: restored.status,
      });
      setSpecimenId(restored.specimen_id ?? "");
      setWeightG(restored.weight_g === null || restored.weight_g === undefined ? "" : String(restored.weight_g));
      setCalibration(restored.calibration);
      setMeasurement(restored.measurement ?? null);
      setStatus(restored.measurement ? "测量完成" : restored.calibration.status === "success" ? "等待测量" : "等待测量");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
      setStatus("需要复核");
    }
  }

  const badgeClass = status === "测量完成" ? "success" : status === "校准失败" || status === "需要复核" ? "error" : "";

  return (
    <>
      <section className="hero">
        <div className="eyebrow">Single Fish Measurement</div>
        <h1>单鱼测量</h1>
        <p>上传单张图片，自动完成校准、测量和正式点位编辑。测量工作台只使用 warped image 和 warped coordinates。</p>
      </section>

      <div className="grid two">
        <div className="card">
          <h2 className="section-title">导入与自动测量</h2>
          <div className="field">
            <label>恢复已有 V2 session（可选）</label>
            <input value={restoreSessionId} placeholder="粘贴 session_id" onChange={(event: ChangeEvent<HTMLInputElement>) => setRestoreSessionId(event.target.value)} />
          </div>
          <div className="button-row" style={{ marginBottom: 16 }}>
            <button className="btn secondary" disabled={!restoreSessionId.trim()} onClick={restoreSession}>恢复会话</button>
            <span className="muted">只读取 V2 session，不读取或修改历史 V1.0 数据。</span>
          </div>
          <div className="field">
            <label>上传图片</label>
            <input type="file" accept="image/*,.heic" onChange={(event: ChangeEvent<HTMLInputElement>) => setFile(event.target.files?.[0] ?? null)} />
          </div>
          <div className="field">
            <label>样本编号 specimen_id</label>
            <input value={specimenId} placeholder="例如 fish_001" onChange={(event: ChangeEvent<HTMLInputElement>) => setSpecimenId(event.target.value)} />
          </div>
          <div className="field">
            <label>样品重量 weight_g（g）</label>
            <input value={weightG} placeholder="例如 155.33" onChange={(event: ChangeEvent<HTMLInputElement>) => setWeightG(event.target.value)} />
          </div>
          <div className="button-row">
            <button className="btn primary" disabled={!file} onClick={runPipeline}>开始测量</button>
            <span className={`status ${badgeClass}`}>{status}</span>
          </div>
          {error && <div className="error-box" style={{ marginTop: 16 }}>{error}</div>}
        </div>

        <div className="card">
          <h2 className="section-title">校准状态</h2>
          <p className="muted">校准失败时后端不会继续测量，也不会把 fallback mm_per_pixel 当作成功。</p>
          {calibration?.raw_marker_overlay_url && (
            <div className="image-box">
              <img src={assetUrl(calibration.raw_marker_overlay_url)} alt="marker overlay" />
            </div>
          )}
          {calibration?.warped_image_url && <p className="success-box">校准成功，warped image 已生成。</p>}
          {calibration && (
            <p className="muted">
              marker: {calibration.detected_marker_count ?? calibration.marker_count} | mm/pixel: {calibration.mm_per_pixel ?? "-"} | warped:{" "}
              {calibration.warped_width_px ?? calibration.warped_image_width ?? "-"} x {calibration.warped_height_px ?? calibration.warped_image_height ?? "-"} |
              canonical warp: {calibration.canonical_warp_used ? "yes" : "no"}
            </p>
          )}
        </div>
      </div>

      {measurement && calibration?.warped_image_url && (
        <div className="grid two" style={{ marginTop: 22 }}>
          <MeasurementCanvas
            imageUrl={assetUrl(calibration.warped_image_url)}
            imageWidth={calibration.warped_image_width ?? 1600}
            imageHeight={calibration.warped_image_height ?? 1000}
            points={measurement.formal_points}
            mmPerPixel={calibration.mm_per_pixel ?? 1}
            onApply={applyPoints}
          />
          <div className="card">
            <h2 className="section-title">测量结果</h2>
            <MetricCards measurements={measurement.measurements} />
            <div className="button-row" style={{ marginTop: 18 }}>
              <button className="btn primary" onClick={doExport}>保存并导出</button>
            </div>
            {exportInfo && <div className="success-box" style={{ marginTop: 16 }}>已导出到：{exportInfo.saved_to}</div>}
          </div>
        </div>
      )}
    </>
  );
}
