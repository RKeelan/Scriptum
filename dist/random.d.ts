/** Sample from standard normal distribution N(0,1). */
export declare function gaussRandom(): number;
/** Sample from bivariate normal distribution. Returns [x, y]. */
export declare function sampleBivariateNormal(mu1: number, mu2: number, sigma1: number, sigma2: number, rho: number): [number, number];
/** Sample an index from a categorical distribution (array of probabilities summing to 1). */
export declare function sampleCategorical(probs: Float32Array): number;
//# sourceMappingURL=random.d.ts.map