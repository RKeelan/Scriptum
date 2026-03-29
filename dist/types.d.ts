/** A named tensor with its data and shape. */
export interface WeightTensor {
    data: Float32Array;
    shape: number[];
}
/** Map from logical weight name to tensor. */
export type WeightDict = Record<string, WeightTensor>;
/** A single stroke point in offset (delta) coordinates. */
export interface StrokePoint {
    dx: number;
    dy: number;
    penUp: boolean;
}
/** An absolute, aligned coordinate in stroke space. */
export interface StrokeCoord {
    x: number;
    y: number;
}
/** Tuning knobs for outline generation. */
export interface OutlineOptions {
    /** Minimum stroke width (at maximum speed). Default 0.1. */
    minWidth?: number;
    /** Maximum stroke width (at zero speed). Default 1.2. */
    maxWidth?: number;
    /** Distance value that maps to minimum width. Default 2.0. */
    speedCap?: number;
    /** Moving-average window size for width smoothing. Default 3. */
    smoothingWindow?: number;
}
/** Options for the synthesize function. */
export interface SynthesizeOptions {
    /** URL or path to the binary weights file. */
    weightsUrl?: string;
    /** Bias parameter controlling neatness (0 = wild, 1 = neat). Default 0.5. */
    bias?: number;
    /** Maximum generation steps. Default 40 * text.length. */
    maxSteps?: number;
}
//# sourceMappingURL=types.d.ts.map