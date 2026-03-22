import { ALPHABET } from "./encoding.js";
import { SynthesisModel } from "./model.js";
import { strokesToSvgPaths } from "./svg.js";
import type { SynthesizeOptions, WeightDict } from "./types.js";
import { fetchWeights } from "./weights.js";

export { ALPHABET, ALPHABET_SIZE } from "./encoding.js";
export { SynthesisModel } from "./model.js";
export { outlinePath, strokesToCoords, strokesToOutlinePaths, strokesToSvgPaths } from "./svg.js";
export type {
  OutlineOptions,
  StrokeCoord,
  StrokePoint,
  SynthesizeOptions,
  WeightDict,
} from "./types.js";
export { fetchWeights, loadWeights } from "./weights.js";

let cachedWeights: WeightDict | null = null;
const DEFAULT_WEIGHTS_URL = "synthesis.bin";

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
export async function synthesize(text: string, options?: SynthesizeOptions): Promise<string[]> {
  // Validate input
  const validChars = new Set(ALPHABET);
  for (const ch of text) {
    if (!validChars.has(ch)) {
      throw new Error(
        `Invalid character '${ch}' (code ${ch.charCodeAt(0)}). ` +
          "Supported characters: letters, digits, and common punctuation.",
      );
    }
  }

  // Load weights if not cached
  if (!cachedWeights) {
    const url = options?.weightsUrl ?? DEFAULT_WEIGHTS_URL;
    cachedWeights = await fetchWeights(url);
  }

  const model = new SynthesisModel(cachedWeights);
  const generateOptions: { bias?: number; maxSteps?: number } = {};
  if (options?.bias !== undefined) generateOptions.bias = options.bias;
  if (options?.maxSteps !== undefined) generateOptions.maxSteps = options.maxSteps;
  const points = model.generate(text, generateOptions);

  return strokesToSvgPaths(points);
}
