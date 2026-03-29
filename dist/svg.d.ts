import type { OutlineOptions, StrokeCoord, StrokePoint } from "./types.js";
/**
 * Convert stroke points (offsets) into SVG path `d` attribute strings.
 *
 * Applies alignment correction and smooth Bezier curves for high-quality output.
 *
 * @returns Array of SVG path `d` strings, one per stroke.
 */
export declare function strokesToSvgPaths(points: StrokePoint[]): string[];
/**
 * Convert stroke points (offsets) into aligned absolute coordinates, split by stroke.
 *
 * Useful for bounding-box computation, animation, and outline generation.
 *
 * @returns Array of coordinate arrays, one per stroke.
 */
export declare function strokesToCoords(points: StrokePoint[]): StrokeCoord[][];
/**
 * Generate a filled outline path for a single stroke with variable width.
 *
 * Width varies with inter-point distance (a velocity proxy), producing
 * natural-looking thick-to-thin transitions. Endpoints taper for pen-on/pen-off.
 *
 * @returns SVG path `d` string for a closed, filled shape.
 */
export declare function outlinePath(stroke: StrokeCoord[], options?: OutlineOptions): string;
/**
 * Convert stroke points into variable-width filled outline paths.
 *
 * Convenience function: runs `strokesToCoords` then `outlinePath` per stroke.
 *
 * @returns Array of SVG path `d` strings (closed, filled shapes), one per stroke.
 */
export declare function strokesToOutlinePaths(points: StrokePoint[], options?: OutlineOptions): string[];
//# sourceMappingURL=svg.d.ts.map