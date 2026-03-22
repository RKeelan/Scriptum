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

/** Options for the synthesize function. */
export interface SynthesizeOptions {
  /** URL or path to the binary weights file. */
  weightsUrl?: string;
  /** Bias parameter controlling neatness (0 = wild, 1 = neat). Default 0.5. */
  bias?: number;
  /** Maximum generation steps. Default 40 * text.length. */
  maxSteps?: number;
}
