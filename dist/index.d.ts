import type { SynthesizeOptions } from "./types.js";
export { ALPHABET, ALPHABET_SIZE } from "./encoding.js";
export { SynthesisModel } from "./model.js";
export { outlinePath, strokesToCoords, strokesToOutlinePaths, strokesToSvgPaths } from "./svg.js";
export type { OutlineOptions, StrokeCoord, StrokePoint, SynthesizeOptions, WeightDict, } from "./types.js";
export { fetchWeights, loadWeights } from "./weights.js";
/**
 * Synthesize handwriting for the given text, returning SVG path `d` strings.
 *
 * Weights are lazily loaded on first call and cached for subsequent calls.
 * Pass `options.weightsUrl` to specify a custom weights location.
 *
 * @param text - Text to render as handwriting
 * @param options - Generation options (weightsUrl, bias, maxSteps)
 * @returns Array of SVG path `d` attribute strings, one per stroke
 */
export declare function synthesize(text: string, options?: SynthesizeOptions): Promise<string[]>;
//# sourceMappingURL=index.d.ts.map