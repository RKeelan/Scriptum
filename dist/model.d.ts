import type { StrokePoint, WeightDict } from "./types.js";
export declare class SynthesisModel {
    private readonly lstm1Kernel;
    private readonly lstm1Bias;
    private readonly lstm2Kernel;
    private readonly lstm2Bias;
    private readonly lstm3Kernel;
    private readonly lstm3Bias;
    private readonly attnWeights;
    private readonly attnBias;
    private readonly gmmWeights;
    private readonly gmmBias;
    constructor(weights: WeightDict);
    /**
     * Generate handwriting strokes for the given text.
     *
     * @param text - Text to synthesize
     * @param options - Generation options
     * @returns Array of stroke points (dx, dy, penUp)
     */
    generate(text: string, options?: {
        bias?: number;
        maxSteps?: number;
    }): StrokePoint[];
}
//# sourceMappingURL=model.d.ts.map