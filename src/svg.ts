import type { StrokePoint } from "./types.js";

/**
 * Convert stroke points (offsets) into SVG path `d` attribute strings.
 *
 * Offsets are converted to absolute coordinates, the y-axis is flipped
 * (handwriting data has y increasing downward, SVG has y increasing downward too,
 * but the model's y convention is inverted), and strokes are split at pen-up events.
 *
 * @returns Array of SVG path `d` strings, one per stroke.
 */
export function strokesToSvgPaths(points: StrokePoint[]): string[] {
  if (points.length === 0) return [];

  // Convert offsets to absolute coordinates
  const coords: { x: number; y: number; penUp: boolean }[] = [];
  let cx = 0;
  let cy = 0;
  for (const p of points) {
    cx += p.dx;
    cy += p.dy;
    coords.push({ x: cx, y: -cy, penUp: p.penUp });
  }

  // Split into strokes at pen-up events
  const paths: string[] = [];
  let currentPath = "";
  let inStroke = false;

  for (const { x, y, penUp } of coords) {
    if (!inStroke) {
      currentPath = `M${x.toFixed(2)},${y.toFixed(2)}`;
      inStroke = true;
    } else {
      currentPath += ` L${x.toFixed(2)},${y.toFixed(2)}`;
    }

    if (penUp) {
      if (currentPath) {
        paths.push(currentPath);
        currentPath = "";
      }
      inStroke = false;
    }
  }

  // Flush any remaining stroke
  if (currentPath) {
    paths.push(currentPath);
  }

  return paths;
}
