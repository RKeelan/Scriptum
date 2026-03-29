import type { WeightDict } from "./types.js";
/**
 * Parse a binary weight file into a WeightDict.
 *
 * Binary format:
 *   Header: magic "SCRP" (4 bytes), tensor_count (uint32 LE)
 *   Per tensor: name_length (uint16 LE), name (utf8), ndims (uint8),
 *               shape (ndims x uint32 LE), data (float32 LE[product(shape)])
 */
export declare function loadWeights(buffer: ArrayBuffer): WeightDict;
/** Fetch a weight file from a URL and parse it. */
export declare function fetchWeights(url: string): Promise<WeightDict>;
//# sourceMappingURL=weights.d.ts.map