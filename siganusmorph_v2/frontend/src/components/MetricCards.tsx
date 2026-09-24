type MetricValue = string | number | null | undefined;

function formatValue(value: MetricValue, suffix = "") {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  if (Number.isFinite(n)) return `${n.toFixed(2)}${suffix}`;
  return `${value}${suffix}`;
}

export function MetricCards({ measurements }: { measurements: Record<string, MetricValue> }) {
  const items = [
    ["压拢尾鳍全长 TL", measurements.TL_compressed_virtual_mm, " mm"],
    ["历史投影 TL", measurements.TL_open_projection_mm, " mm"],
    ["标准长 SL", measurements.SL_mm, " mm"],
    ["叉长 FL", measurements.FL_mm, " mm"],
    ["体高", measurements.body_depth_mm, " mm"],
    ["尾柄高", measurements.caudal_peduncle_depth_mm, " mm"],
  ] as const;
  return (
    <div className="metrics">
      {items.map(([label, value, suffix]) => (
        <div className="metric" key={label}>
          <div className="metric-label">{label}</div>
          <div className="metric-value">{formatValue(value, suffix)}</div>
        </div>
      ))}
    </div>
  );
}
