export const API_BASE_URL = "http://127.0.0.1:8000";
export const API_BASE = API_BASE_URL;

export type HealthResponse = {
  status: "ok";
  version: string;
  app: string;
};

export type SettingsResponse = {
  version: string;
  app: string;
  core_engine: string;
  backend_output_root: string;
  user_output_root: string;
  admin_output_root: string;
  model_paths: Array<{
    name: string;
    path: string;
    exists: boolean;
  }>;
  notes: string[];
};

export type UploadResponse = {
  version: string;
  session_id: string;
  image_name: string;
  raw_image_url: string;
  status: string;
};

export type CalibrateResponse = {
  version: string;
  session_id: string;
  status: string;
  calibration_mode?: string;
  warped_image_url?: string | null;
  raw_marker_overlay_url?: string | null;
  warped_image_width?: number | null;
  warped_image_height?: number | null;
  mm_per_pixel?: number | null;
  mm_per_pixel_source?: string;
  marker_count: number;
  message: string;
  calibration_status?: string;
  detected_marker_count?: number;
  homography_raw_to_warped?: number[][] | null;
  detected_markers?: unknown[];
  detected_board_corners_raw?: unknown[];
  warped_width_px?: number | null;
  warped_height_px?: number | null;
  board_physical_width_mm?: number | null;
  board_physical_height_mm?: number | null;
  canonical_warp_used?: boolean;
  measurement_roi_bounds_warped?: number[] | null;
  fish_bbox_warped?: number[] | null;
  fish_margin_to_roi_px?: Record<string, number> | null;
  coordinate_system_version?: string;
};

export type MeasureResponse = {
  version: string;
  session_id: string;
  coordinate_space: "warped_image";
  formal_points: Record<string, [number, number]>;
  derived_points: Record<string, unknown>;
  measurements: Record<string, number | string | null>;
  qc: Record<string, unknown>;
  status: string;
};

export type ExportResponse = {
  version: string;
  session_id: string;
  files: Record<string, string>;
  saved_to: string;
};

export type ExportHistoryEntry = ExportResponse & {
  image_name?: string;
  specimen_id?: string;
  weight_g?: string | number | null;
  exported_at: string | number;
};

export type ExportListResponse = {
  version: string;
  exports: ExportHistoryEntry[];
};

export type SessionResponse = {
  version: string;
  session_id: string;
  image_name: string;
  specimen_id: string;
  weight_g?: number | string | null;
  status: string;
  coordinate_space: "warped_image";
  raw_image_url?: string | null;
  raw_image_width?: number | null;
  raw_image_height?: number | null;
  calibration: CalibrateResponse;
  measurement?: MeasureResponse | null;
  exports: Record<string, string>;
  created_at?: string;
  updated_at?: string;
};

const EXPORT_HISTORY_KEY = "siganusmorph_v2_export_history";

function safeErrorMessage(detail: string): string {
  try {
    const parsed = JSON.parse(detail) as { detail?: unknown };
    if (typeof parsed.detail === "string") return parsed.detail;
  } catch {
    // Fall back to normalized text below.
  }
  if (detail.includes("Traceback") || detail.includes('File "')) {
    return "处理失败。请检查图像质量、校准状态，或联系管理员复核。";
  }
  return detail || "请求失败。";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(safeErrorMessage(detail) || `API request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export async function getSettings(): Promise<SettingsResponse> {
  return request<SettingsResponse>("/api/settings");
}

export async function uploadImage(file: File, specimenId: string, weightG: string): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("specimen_id", specimenId);
  if (weightG.trim()) form.append("weight_g", weightG);
  return request<UploadResponse>("/api/upload", { method: "POST", body: form });
}

export async function getSession(sessionId: string): Promise<SessionResponse> {
  return request<SessionResponse>(`/api/session/${encodeURIComponent(sessionId)}`);
}

export async function calibrate(sessionId: string, manualCorners?: number[][]): Promise<CalibrateResponse> {
  return request<CalibrateResponse>("/api/calibrate", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, manual_corners: manualCorners ?? null }),
  });
}

export function rememberExport(entry: ExportHistoryEntry): void {
  const history = loadExportHistory();
  const next = [entry, ...history.filter((item) => item.session_id !== entry.session_id)].slice(0, 100);
  window.localStorage.setItem(EXPORT_HISTORY_KEY, JSON.stringify(next));
}

export function loadExportHistory(): ExportHistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(EXPORT_HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as ExportHistoryEntry[]) : [];
  } catch {
    return [];
  }
}

export function clearExportHistory(): void {
  window.localStorage.removeItem(EXPORT_HISTORY_KEY);
}

export async function measure(sessionId: string): Promise<MeasureResponse> {
  return request<MeasureResponse>("/api/measure", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      preannotation_mode: "recommended",
      measurement_axis_mode: "auto_qc_gated",
    }),
  });
}

export async function updatePoints(
  sessionId: string,
  formalPoints: Record<string, [number, number]>,
): Promise<MeasureResponse> {
  return request<MeasureResponse>("/api/update-points", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      formal_points: formalPoints,
      measurement_overrides: {
        source: "react_measurement_canvas",
        modified_at: new Date().toISOString(),
      },
    }),
  });
}

export async function exportSession(sessionId: string): Promise<ExportResponse> {
  return request<ExportResponse>("/api/export", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      formats: ["csv", "xlsx", "json", "preview_png"],
      save_confirmed_session: true,
    }),
  });
}

export async function listExports(): Promise<ExportListResponse> {
  return request<ExportListResponse>("/api/exports");
}

export function assetUrl(path?: string | null): string {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}
