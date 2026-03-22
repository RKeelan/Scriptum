import { expVec, softmax, tanhVec } from "./math.js";
import { sampleBivariateNormal, sampleCategorical } from "./random.js";

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
export function parseMdnOutput(raw: Float32Array, numComponents: number, bias: number): MdnParams {
  const N = numComponents;

  // Split according to TF implementation order: pis, sigmas, rhos, mus, es
  const rawPis = raw.slice(0, N);
  const rawSigmas = raw.slice(N, 3 * N);
  const rawRhos = raw.slice(3 * N, 4 * N);
  const rawMus = raw.slice(4 * N, 6 * N);
  const rawEs = raw.slice(6 * N, 6 * N + 1);

  // Apply bias to pis and sigmas before activation
  const biasedPis = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    biasedPis[i] = (rawPis[i] as number) * (1 + bias);
  }
  const biasedSigmas = new Float32Array(2 * N);
  for (let i = 0; i < 2 * N; i++) {
    biasedSigmas[i] = (rawSigmas[i] as number) - bias;
  }

  // Activations
  const pis = softmax(biasedPis);
  const sigmas = expVec(biasedSigmas);
  const rhos = tanhVec(rawRhos);
  const esVal = 1 / (1 + Math.exp(-(rawEs[0] as number)));

  // Clip sigmas to minimum 1e-4
  for (let i = 0; i < sigmas.length; i++) {
    if ((sigmas[i] as number) < 1e-4) sigmas[i] = 1e-4;
  }

  // Clip rhos to (-1+eps, 1-eps)
  const eps = 1e-8;
  for (let i = 0; i < rhos.length; i++) {
    const v = rhos[i] as number;
    if (v < eps - 1) rhos[i] = eps - 1;
    else if (v > 1 - eps) rhos[i] = 1 - eps;
  }

  // Clip eos
  const eos = Math.max(eps, Math.min(1 - eps, esVal));

  // Split mus and sigmas into x and y components
  const mu1 = rawMus.slice(0, N);
  const mu2 = rawMus.slice(N, 2 * N);
  const sigma1 = sigmas.slice(0, N);
  const sigma2 = sigmas.slice(N, 2 * N);

  // Zero out very small pis (matching TF: pis < 0.01 → 0)
  for (let i = 0; i < N; i++) {
    if ((pis[i] as number) < 0.01) pis[i] = 0;
  }

  return { pis, mu1, mu2, sigma1, sigma2, rho: rhos, eos };
}

/**
 * Sample a stroke point from the MDN mixture distribution.
 * @returns [dx, dy, penUp] tuple
 */
export function sampleFromMixture(params: MdnParams): [number, number, boolean] {
  const idx = sampleCategorical(params.pis);
  const [dx, dy] = sampleBivariateNormal(
    params.mu1[idx] as number,
    params.mu2[idx] as number,
    params.sigma1[idx] as number,
    params.sigma2[idx] as number,
    params.rho[idx] as number,
  );
  const penUp = Math.random() < params.eos;
  return [dx, dy, penUp];
}
