/** The 73-character alphabet matching the IAM training data. Index 0 is end-of-sequence. */
export declare const ALPHABET: readonly string[];
export declare const ALPHABET_SIZE: number;
/** Encode a string as an array of character indices, with a trailing end-of-sequence (0). */
export declare function encodeText(text: string): number[];
/** Create a one-hot Float32Array of the given size with a 1 at the given index. */
export declare function oneHot(index: number, size: number): Float32Array;
//# sourceMappingURL=encoding.d.ts.map