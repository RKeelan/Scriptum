/** Parsed MDN parameters for one timestep. */
export interface MdnParams {
    pis: Float32Array;
    mu1: Float32Array;
    mu2: Float32Array;
    sigma1: Float32Array;
    sigma2: Float32Array;
    rho: Float32Array;
    eos: number;
}
/**
 * Parse raw MDN output (121 values) into mixture parameters.
 *
 * Parameter layout (matching TF implementation):
 *   pis[20], sigmas[40], rhos[20], mus[40], es[1]
 *
 * @param raw - Raw output from the GMM dense layer [121]
 * @param numComponents - Number of mixture components (20)
 * @param bias - Bias parameter for controlling neatness (0 = wild, higher = neater)
 */
export declare function parseMdnOutput(raw: Float32Array, numComponents: number, bias: number): MdnParams;
/**
 * Sample a stroke point from the MDN mixture distribution.
 * @returns [dx, dy, penUp] tuple
 */
export declare function sampleFromMixture(params: MdnParams): [number, number, boolean];
//# sourceMappingURL=mdn.d.ts.map