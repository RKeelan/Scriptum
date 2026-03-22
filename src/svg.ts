import type { OutlineOptions, StrokeCoord, StrokePoint } from "./types.js";

/**
 * Compute a simple linear regression to find the baseline slant and correct it.
 * Adapted from the reference Python implementation's `align` function.
 */
function alignCoords(coords: StrokeCoord[]): StrokeCoord[] {
  if (coords.length < 2) return coords;

  // Linear regression: y = offset + slope * x
  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumXY = 0;
  const n = coords.length;

  for (const c of coords) {
    sumX += c.x;
    sumY += c.y;
    sumXX += c.x * c.x;
    sumXY += c.x * c.y;
  }

  const denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 1e-10) return coords;

  const slope = (n * sumXY - sumX * sumY) / denom;
  const offset = (sumY - slope * sumX) / n;
  const theta = Math.atan(slope);
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);

  return coords.map((c) => ({
    x: cosT * c.x + sinT * c.y,
    y: -sinT * c.x + cosT * c.y - offset,
  }));
}

/** Convert offset-based stroke points to absolute coordinates with y-negation. */
function convertAndAlign(points: StrokePoint[]): { x: number; y: number; penUp: boolean }[] {
  const allCoords: { x: number; y: number; penUp: boolean }[] = [];
  let cx = 0;
  let cy = 0;
  for (const p of points) {
    cx += p.dx;
    cy += p.dy;
    allCoords.push({ x: cx, y: -cy, penUp: p.penUp });
  }

  const xyOnly = allCoords.map((c) => ({ x: c.x, y: c.y }));
  const aligned = alignCoords(xyOnly);
  for (let i = 0; i < allCoords.length; i++) {
    const a = aligned[i] as StrokeCoord;
    const c = allCoords[i] as { x: number; y: number; penUp: boolean };
    c.x = a.x;
    c.y = a.y;
  }

  return allCoords;
}

/** Split aligned coordinates into separate strokes at pen-up boundaries. */
function splitStrokes(allCoords: { x: number; y: number; penUp: boolean }[]): StrokeCoord[][] {
  const strokes: StrokeCoord[][] = [];
  let current: StrokeCoord[] = [];

  for (const { x, y, penUp } of allCoords) {
    current.push({ x, y });
    if (penUp) {
      if (current.length > 0) {
        strokes.push(current);
        current = [];
      }
    }
  }

  if (current.length > 0) {
    strokes.push(current);
  }

  return strokes;
}

/**
 * Build a smooth SVG path from a sequence of points using quadratic Bezier curves.
 * Each segment uses the midpoint between consecutive points as the curve endpoint,
 * with the actual point as the control point.
 */
function smoothPath(points: StrokeCoord[]): string {
  if (points.length === 0) return "";
  if (points.length === 1) {
    const p = points[0] as StrokeCoord;
    return `M${p.x.toFixed(2)},${p.y.toFixed(2)}`;
  }
  if (points.length === 2) {
    const p0 = points[0] as StrokeCoord;
    const p1 = points[1] as StrokeCoord;
    return `M${p0.x.toFixed(2)},${p0.y.toFixed(2)} L${p1.x.toFixed(2)},${p1.y.toFixed(2)}`;
  }

  const first = points[0] as StrokeCoord;
  let d = `M${first.x.toFixed(2)},${first.y.toFixed(2)}`;

  // Line to midpoint of first segment
  const p1 = points[1] as StrokeCoord;
  const mx = (first.x + p1.x) / 2;
  const my = (first.y + p1.y) / 2;
  d += ` L${mx.toFixed(2)},${my.toFixed(2)}`;

  // Quadratic Bezier through subsequent points
  for (let i = 1; i < points.length - 1; i++) {
    const cp = points[i] as StrokeCoord;
    const next = points[i + 1] as StrokeCoord;
    const endX = (cp.x + next.x) / 2;
    const endY = (cp.y + next.y) / 2;
    d += ` Q${cp.x.toFixed(2)},${cp.y.toFixed(2)} ${endX.toFixed(2)},${endY.toFixed(2)}`;
  }

  // Line to last point
  const last = points[points.length - 1] as StrokeCoord;
  d += ` L${last.x.toFixed(2)},${last.y.toFixed(2)}`;

  return d;
}

/**
 * Convert stroke points (offsets) into SVG path `d` attribute strings.
 *
 * Applies alignment correction and smooth Bezier curves for high-quality output.
 *
 * @returns Array of SVG path `d` strings, one per stroke.
 */
export function strokesToSvgPaths(points: StrokePoint[]): string[] {
  if (points.length === 0) return [];

  const allCoords = convertAndAlign(points);
  const strokes = splitStrokes(allCoords);
  return strokes.map(smoothPath);
}

/**
 * Convert stroke points (offsets) into aligned absolute coordinates, split by stroke.
 *
 * Useful for bounding-box computation, animation, and outline generation.
 *
 * @returns Array of coordinate arrays, one per stroke.
 */
export function strokesToCoords(points: StrokePoint[]): StrokeCoord[][] {
  if (points.length === 0) return [];

  const allCoords = convertAndAlign(points);
  return splitStrokes(allCoords);
}

const DEFAULT_MIN_WIDTH = 0.1;
const DEFAULT_MAX_WIDTH = 1.2;
const DEFAULT_SPEED_CAP = 2.0;
const DEFAULT_SMOOTHING_WINDOW = 3;

/**
 * Generate a filled outline path for a single stroke with variable width.
 *
 * Width varies with inter-point distance (a velocity proxy), producing
 * natural-looking thick-to-thin transitions. Endpoints taper for pen-on/pen-off.
 *
 * @returns SVG path `d` string for a closed, filled shape.
 */
export function outlinePath(stroke: StrokeCoord[], options?: OutlineOptions): string {
  if (stroke.length === 0) return "";

  if (stroke.length === 1) {
    // Draw a small circle for single-point strokes
    const p = stroke[0] as StrokeCoord;
    const r = ((options?.maxWidth ?? DEFAULT_MAX_WIDTH) / 2) * 0.3;
    return (
      `M${(p.x - r).toFixed(2)},${p.y.toFixed(2)} ` +
      `A${r.toFixed(2)},${r.toFixed(2)} 0 1,0 ${(p.x + r).toFixed(2)},${p.y.toFixed(2)} ` +
      `A${r.toFixed(2)},${r.toFixed(2)} 0 1,0 ${(p.x - r).toFixed(2)},${p.y.toFixed(2)}Z`
    );
  }

  const minW = options?.minWidth ?? DEFAULT_MIN_WIDTH;
  const maxW = options?.maxWidth ?? DEFAULT_MAX_WIDTH;
  const speedCap = options?.speedCap ?? DEFAULT_SPEED_CAP;
  const smoothWin = options?.smoothingWindow ?? DEFAULT_SMOOTHING_WINDOW;

  const n = stroke.length;

  // 1. Compute inter-point distances (velocity proxy)
  const distances: number[] = [0];
  for (let i = 1; i < n; i++) {
    const prev = stroke[i - 1] as StrokeCoord;
    const curr = stroke[i] as StrokeCoord;
    const dx = curr.x - prev.x;
    const dy = curr.y - prev.y;
    distances.push(Math.sqrt(dx * dx + dy * dy));
  }

  // 2. Map velocity to width
  const rawWidths: number[] = distances.map((d) => {
    const t = Math.min(d / speedCap, 1);
    return maxW + t * (minW - maxW); // lerp(maxW, minW, t)
  });

  // 3. Smooth widths with moving average
  const widths: number[] = [];
  const halfWin = Math.floor(smoothWin / 2);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - halfWin); j <= Math.min(n - 1, i + halfWin); j++) {
      sum += rawWidths[j] as number;
      count++;
    }
    widths.push(sum / count);
  }

  // 4. Taper first/last points toward zero
  const taperLen = Math.min(3, Math.floor(n / 2));
  for (let i = 0; i < taperLen; i++) {
    const t = (i + 1) / (taperLen + 1);
    widths[i] = (widths[i] as number) * t;
    widths[n - 1 - i] = (widths[n - 1 - i] as number) * t;
  }

  // 5. Compute normals at each point
  const normals: StrokeCoord[] = [];
  for (let i = 0; i < n; i++) {
    let tx: number;
    let ty: number;
    if (i === 0) {
      const next = stroke[1] as StrokeCoord;
      const curr = stroke[0] as StrokeCoord;
      tx = next.x - curr.x;
      ty = next.y - curr.y;
    } else if (i === n - 1) {
      const curr = stroke[n - 1] as StrokeCoord;
      const prev = stroke[n - 2] as StrokeCoord;
      tx = curr.x - prev.x;
      ty = curr.y - prev.y;
    } else {
      const next = stroke[i + 1] as StrokeCoord;
      const prev = stroke[i - 1] as StrokeCoord;
      tx = next.x - prev.x;
      ty = next.y - prev.y;
    }

    const len = Math.sqrt(tx * tx + ty * ty);
    if (len < 1e-10) {
      normals.push({ x: 0, y: 1 });
    } else {
      normals.push({ x: -ty / len, y: tx / len });
    }
  }

  // 6. Offset left/right by local half-width
  const left: StrokeCoord[] = [];
  const right: StrokeCoord[] = [];
  for (let i = 0; i < n; i++) {
    const p = stroke[i] as StrokeCoord;
    const nm = normals[i] as StrokeCoord;
    const hw = (widths[i] as number) / 2;
    left.push({ x: p.x + nm.x * hw, y: p.y + nm.y * hw });
    right.push({ x: p.x - nm.x * hw, y: p.y - nm.y * hw });
  }

  // 7. Build closed path: forward along left edge, reverse along right edge
  let d = `M${(left[0] as StrokeCoord).x.toFixed(2)},${(left[0] as StrokeCoord).y.toFixed(2)}`;

  // Left edge (forward) — quadratic Bezier smoothing
  if (left.length === 2) {
    const p = left[1] as StrokeCoord;
    d += ` L${p.x.toFixed(2)},${p.y.toFixed(2)}`;
  } else {
    for (let i = 1; i < left.length - 1; i++) {
      const cp = left[i] as StrokeCoord;
      const next = left[i + 1] as StrokeCoord;
      const endX = (cp.x + next.x) / 2;
      const endY = (cp.y + next.y) / 2;
      if (i === 1) {
        // Line to first midpoint
        const prev = left[0] as StrokeCoord;
        const fmx = (prev.x + cp.x) / 2;
        const fmy = (prev.y + cp.y) / 2;
        d += ` L${fmx.toFixed(2)},${fmy.toFixed(2)}`;
      }
      d += ` Q${cp.x.toFixed(2)},${cp.y.toFixed(2)} ${endX.toFixed(2)},${endY.toFixed(2)}`;
    }
    const lastL = left[left.length - 1] as StrokeCoord;
    d += ` L${lastL.x.toFixed(2)},${lastL.y.toFixed(2)}`;
  }

  // Cap: line from left end to right end
  const lastR = right[right.length - 1] as StrokeCoord;
  d += ` L${lastR.x.toFixed(2)},${lastR.y.toFixed(2)}`;

  // Right edge (reverse) — quadratic Bezier smoothing
  if (right.length === 2) {
    const p = right[0] as StrokeCoord;
    d += ` L${p.x.toFixed(2)},${p.y.toFixed(2)}`;
  } else {
    for (let i = right.length - 2; i > 0; i--) {
      const cp = right[i] as StrokeCoord;
      const next = right[i - 1] as StrokeCoord;
      const endX = (cp.x + next.x) / 2;
      const endY = (cp.y + next.y) / 2;
      if (i === right.length - 2) {
        // Line to first midpoint (from end)
        const prev = right[right.length - 1] as StrokeCoord;
        const fmx = (prev.x + cp.x) / 2;
        const fmy = (prev.y + cp.y) / 2;
        d += ` L${fmx.toFixed(2)},${fmy.toFixed(2)}`;
      }
      d += ` Q${cp.x.toFixed(2)},${cp.y.toFixed(2)} ${endX.toFixed(2)},${endY.toFixed(2)}`;
    }
    const firstR = right[0] as StrokeCoord;
    d += ` L${firstR.x.toFixed(2)},${firstR.y.toFixed(2)}`;
  }

  d += "Z";
  return d;
}

/**
 * Convert stroke points into variable-width filled outline paths.
 *
 * Convenience function: runs `strokesToCoords` then `outlinePath` per stroke.
 *
 * @returns Array of SVG path `d` strings (closed, filled shapes), one per stroke.
 */
export function strokesToOutlinePaths(points: StrokePoint[], options?: OutlineOptions): string[] {
  const strokes = strokesToCoords(points);
  return strokes.map((s) => outlinePath(s, options));
}
