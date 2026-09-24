import { ChangeEvent, KeyboardEvent, PointerEvent, useEffect, useMemo, useRef, useState } from "react";

export type PointMap = Record<string, [number, number]>;

const DRAGGABLE = [
  "P1_snout_tip",
  "P2_eye_anterior",
  "P3_operculum_posterior",
  "P4_peduncle_anterior",
  "P5_caudal_base",
  "P6_caudal_fork",
  "P7U_upper_lobe_tip",
  "P7L_lower_lobe_tip",
  "body_depth_upper",
  "body_depth_lower",
  "peduncle_depth_upper",
  "peduncle_depth_lower",
];

const COLORS: Record<string, string> = {
  P1_snout_tip: "#ffcc00",
  P2_eye_anterior: "#ffcc00",
  P3_operculum_posterior: "#ffcc00",
  P4_peduncle_anterior: "#ffcc00",
  P5_caudal_base: "#ffcc00",
  P6_caudal_fork: "#ffcc00",
  P7U_upper_lobe_tip: "#ffcc00",
  P7L_lower_lobe_tip: "#ffcc00",
  body_depth_upper: "#0071e3",
  body_depth_lower: "#0071e3",
  peduncle_depth_upper: "#00a6a6",
  peduncle_depth_lower: "#00a6a6",
};

const VISIBLE_RADIUS_NORMAL = 7;
const VISIBLE_RADIUS_DEPTH = 8;
const VISIBLE_RADIUS_ACTIVE = 11;
const HIT_RADIUS_DISPLAY_PX = 24;

function dist(a?: [number, number], b?: [number, number]) {
  if (!a || !b) return null;
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function deriveP7V(points: PointMap): [number, number] | null {
  const p5 = points.P5_caudal_base;
  const u = points.P7U_upper_lobe_tip;
  const l = points.P7L_lower_lobe_tip;
  if (!p5 || !u || !l) return null;
  const mid: [number, number] = [(u[0] + l[0]) / 2, (u[1] + l[1]) / 2];
  const axis = [mid[0] - p5[0], mid[1] - p5[1]];
  const axisLen = Math.hypot(axis[0], axis[1]);
  if (axisLen < 1e-6) return null;
  const unit = [axis[0] / axisLen, axis[1] / axisLen];
  const r = Math.max(Math.hypot(u[0] - p5[0], u[1] - p5[1]), Math.hypot(l[0] - p5[0], l[1] - p5[1]));
  return [p5[0] + unit[0] * r, p5[1] + unit[1] * r];
}

function localMeasurements(points: PointMap, mmPerPixel: number) {
  const p7v = deriveP7V(points);
  const sl = dist(points.P1_snout_tip, points.P5_caudal_base);
  const fl = dist(points.P1_snout_tip, points.P6_caudal_fork);
  const tlComp = p7v && points.P1_snout_tip ? dist(points.P1_snout_tip, p7v) : null;
  const bodyDepth = dist(points.body_depth_upper, points.body_depth_lower);
  const peduncleDepth = dist(points.peduncle_depth_upper, points.peduncle_depth_lower);
  return {
    TL_compressed_virtual_mm: tlComp === null ? null : tlComp * mmPerPixel,
    SL_mm: sl === null ? null : sl * mmPerPixel,
    FL_mm: fl === null ? null : fl * mmPerPixel,
    body_depth_mm: bodyDepth === null ? null : bodyDepth * mmPerPixel,
    caudal_peduncle_depth_mm: peduncleDepth === null ? null : peduncleDepth * mmPerPixel,
  };
}

export function imageToDisplay(xy: [number, number], svg: SVGSVGElement): [number, number] {
  const point = svg.createSVGPoint();
  point.x = xy[0];
  point.y = xy[1];
  const matrix = svg.getScreenCTM();
  if (!matrix) {
    const rect = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    return [
      rect.left + ((xy[0] - viewBox.x) / viewBox.width) * rect.width,
      rect.top + ((xy[1] - viewBox.y) / viewBox.height) * rect.height,
    ];
  }
  const mapped = point.matrixTransform(matrix);
  return [mapped.x, mapped.y];
}

export function displayToImage(xy: [number, number], svg: SVGSVGElement): [number, number] {
  const point = svg.createSVGPoint();
  point.x = xy[0];
  point.y = xy[1];
  const matrix = svg.getScreenCTM();
  if (!matrix) {
    const rect = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    return [
      viewBox.x + ((xy[0] - rect.left) / rect.width) * viewBox.width,
      viewBox.y + ((xy[1] - rect.top) / rect.height) * viewBox.height,
    ];
  }
  const mapped = point.matrixTransform(matrix.inverse());
  return [mapped.x, mapped.y];
}

function clampToImage(xy: [number, number], width: number, height: number): [number, number] {
  return [Math.max(0, Math.min(width, xy[0])), Math.max(0, Math.min(height, xy[1]))];
}

export function MeasurementCanvas({
  imageUrl,
  imageWidth,
  imageHeight,
  points,
  mmPerPixel,
  onApply,
}: {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  points: PointMap;
  mmPerPixel: number;
  onApply: (points: PointMap, local: Record<string, number | null>) => void;
}) {
  const [localPoints, setLocalPoints] = useState<PointMap>(points);
  const [active, setActive] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [pointer, setPointer] = useState<[number, number] | null>(null);
  const [showAllLabels, setShowAllLabels] = useState(false);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const p7v = deriveP7V(localPoints);
  const measurements = useMemo(() => localMeasurements(localPoints, mmPerPixel), [localPoints, mmPerPixel]);

  useEffect(() => {
    setLocalPoints(points);
  }, [points]);

  function eventToImage(event: PointerEvent<SVGSVGElement>): [number, number] {
    return clampToImage(displayToImage([event.clientX, event.clientY], event.currentTarget), imageWidth, imageHeight);
  }

  function nearestPoint(xy: [number, number], svg: SVGSVGElement): string | null {
    const pointerDisplay = imageToDisplay(xy, svg);
    let best: string | null = null;
    let bestDist = Infinity;
    for (const name of DRAGGABLE) {
      const p = localPoints[name];
      if (!p) continue;
      const displayPoint = imageToDisplay(p, svg);
      const d = Math.hypot(displayPoint[0] - pointerDisplay[0], displayPoint[1] - pointerDisplay[1]);
      if (d < HIT_RADIUS_DISPLAY_PX && d < bestDist) {
        best = name;
        bestDist = d;
      }
    }
    return best;
  }

  function handlePointerDown(event: PointerEvent<SVGSVGElement>) {
    const xy = eventToImage(event);
    const target = nearestPoint(xy, event.currentTarget);
    if (target) {
      setActive(target);
      event.currentTarget.setPointerCapture(event.pointerId);
    }
  }

  function handlePointerMove(event: PointerEvent<SVGSVGElement>) {
    const xy = eventToImage(event);
    setPointer(xy);
    const target = nearestPoint(xy, event.currentTarget);
    setHover(target);
    if (active) {
      setLocalPoints((prev) => ({ ...prev, [active]: xy }));
    }
  }

  function handlePointerUp(event: PointerEvent<SVGSVGElement>) {
    setActive(null);
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // Pointer may already be released by the browser.
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (!active && !hover) return;
    const name = active ?? hover;
    if (!name) return;
    const step = event.shiftKey ? 5 : 1;
    const delta: Record<string, [number, number]> = {
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
    };
    const d = delta[event.key];
    if (!d) return;
    event.preventDefault();
    setLocalPoints((prev) => {
      const p = prev[name];
      if (!p) return prev;
      return { ...prev, [name]: clampToImage([p[0] + d[0], p[1] + d[1]], imageWidth, imageHeight) };
    });
  }

  const bodyLine = localPoints.body_depth_upper && localPoints.body_depth_lower ? [localPoints.body_depth_upper, localPoints.body_depth_lower] : null;
  const pedLine =
    localPoints.peduncle_depth_upper && localPoints.peduncle_depth_lower
      ? [localPoints.peduncle_depth_upper, localPoints.peduncle_depth_lower]
      : null;

  return (
    <div className="card" onKeyDown={handleKeyDown} tabIndex={0}>
      <div className="canvas-header">
        <div>
          <h2 className="section-title">Measurement Canvas</h2>
          <p className="muted">Drag points locally, then apply changes to update backend measurements.</p>
        </div>
        <label className="toggle-row">
          <input type="checkbox" checked={showAllLabels} onChange={(event: ChangeEvent<HTMLInputElement>) => setShowAllLabels(event.target.checked)} />
          Show labels
        </label>
      </div>
      <div className="canvas-wrap">
        <svg
          ref={svgRef}
          className="measurement-svg"
          viewBox={`0 0 ${imageWidth} ${imageHeight}`}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={() => {
            setPointer(null);
            setHover(null);
          }}
        >
          <image href={imageUrl} width={imageWidth} height={imageHeight} preserveAspectRatio="xMidYMid meet" />
          {bodyLine && (
            <line
              x1={bodyLine[0][0]}
              y1={bodyLine[0][1]}
              x2={bodyLine[1][0]}
              y2={bodyLine[1][1]}
              stroke="#0071e3"
              strokeWidth="5"
              strokeLinecap="round"
            />
          )}
          {pedLine && (
            <line
              x1={pedLine[0][0]}
              y1={pedLine[0][1]}
              x2={pedLine[1][0]}
              y2={pedLine[1][1]}
              stroke="#00a6a6"
              strokeWidth="5"
              strokeLinecap="round"
            />
          )}
          {p7v && (
            <g className="derived-point">
              <circle cx={p7v[0]} cy={p7v[1]} r="13" fill="white" opacity="0.92" />
              <polygon
                points={`${p7v[0]},${p7v[1] - 9} ${p7v[0] + 9},${p7v[1]} ${p7v[0]},${p7v[1] + 9} ${p7v[0] - 9},${p7v[1]}`}
                fill="none"
                stroke="#af52de"
                strokeWidth="3"
              />
              <text x={p7v[0] + 13} y={p7v[1] - 13} fontSize="16" fill="#1d1d1f" stroke="white" strokeWidth="4" paintOrder="stroke">
                P7V derived
              </text>
            </g>
          )}
          {DRAGGABLE.map((name) => {
            const p = localPoints[name];
            if (!p) return null;
            const color = COLORS[name] ?? "#ffd60a";
            const r = active === name || hover === name ? VISIBLE_RADIUS_ACTIVE : name.includes("depth") ? VISIBLE_RADIUS_DEPTH : VISIBLE_RADIUS_NORMAL;
            const labelVisible = showAllLabels || hover === name || active === name;
            return (
              <g key={name} className={active === name ? "point active" : hover === name ? "point hover" : "point"}>
                <circle cx={p[0]} cy={p[1]} r={r + 4} fill="white" opacity="0.95" />
                <circle cx={p[0]} cy={p[1]} r={r} fill={color} stroke="#111827" strokeWidth="2.2" />
                {labelVisible && (
                  <text x={p[0] + 13} y={p[1] - 13} fontSize="16" fill="#1d1d1f" stroke="white" strokeWidth="4" paintOrder="stroke">
                    {name} x={p[0].toFixed(1)}, y={p[1].toFixed(1)}
                  </text>
                )}
              </g>
            );
          })}
          {pointer && active && (
            <g className="crosshair">
              <line x1="0" y1={pointer[1]} x2={imageWidth} y2={pointer[1]} stroke="#0071e3" strokeWidth="1.5" opacity="0.65" />
              <line x1={pointer[0]} y1="0" x2={pointer[0]} y2={imageHeight} stroke="#0071e3" strokeWidth="1.5" opacity="0.65" />
            </g>
          )}
        </svg>
        {pointer && (
          <div
            className="magnifier"
            style={{
              left: "18px",
              bottom: "18px",
              backgroundImage: `url(${imageUrl})`,
              backgroundRepeat: "no-repeat",
              backgroundSize: `${imageWidth * 2.5}px ${imageHeight * 2.5}px`,
              backgroundPosition: `${90 - pointer[0] * 2.5}px ${90 - pointer[1] * 2.5}px`,
            }}
          />
        )}
      </div>
      <div className="button-row" style={{ marginTop: 16 }}>
        <button className="btn primary" onClick={() => onApply(localPoints, measurements)}>
          Apply changes
        </button>
        <span className="muted">Arrow keys nudge 1 px; Shift + arrows nudge 5 px. P7V is derived and not directly draggable.</span>
      </div>
      <div className="metrics" style={{ marginTop: 16 }}>
        {Object.entries(measurements).map(([key, value]) => (
          <div className="metric" key={key}>
            <div className="metric-label">{key}</div>
            <div className="metric-value">{value === null ? "-" : value.toFixed(2)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
