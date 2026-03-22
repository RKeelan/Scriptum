import { concat, matVecMul, softplus, vecAdd } from "./math.js";

/**
 * Compute one step of the soft Gaussian attention window.
 *
 * The attention mechanism produces a context window over the character sequence
 * by combining K Gaussian components. The kappa parameter is cumulative
 * (monotonically increasing), ensuring left-to-right reading.
 *
 * @param prevWindow - Previous attention window vector [alphabetSize]
 * @param x - Current input [3] (dx, dy, penUp)
 * @param h1 - Hidden state from LSTM layer 1 [numUnits]
 * @param kappaPrev - Previous kappa values [K]
 * @param charOneHots - One-hot encoded character sequence, flat [charLen * alphabetSize]
 * @param charLen - Number of characters in the sequence
 * @param alphabetSize - Size of the character alphabet
 * @param weights - Attention dense layer weights [3K, inputSize] row-major
 * @param bias - Attention dense layer bias [3K]
 * @param K - Number of attention mixture components
 * @returns [newWindow, newKappa] — new context window and updated kappa
 */
export function attentionStep(
  prevWindow: Float32Array,
  x: Float32Array,
  h1: Float32Array,
  kappaPrev: Float32Array,
  charOneHots: Float32Array,
  charLen: number,
  alphabetSize: number,
  weights: Float32Array,
  bias: Float32Array,
  K: number,
): [Float32Array, Float32Array] {
  // Dense layer input: concat(prevWindow, x, h1)
  const input = concat(prevWindow, x, h1);
  const inputSize = input.length;
  const outputSize = 3 * K;

  // params = weights * input + bias, then softplus
  const raw = vecAdd(matVecMul(weights, input, outputSize, inputSize), bias);
  const params = softplus(raw);

  // Split into alpha, beta, kappa_delta (each length K)
  const alpha = params.subarray(0, K);
  const beta = params.subarray(K, 2 * K);
  const kappaDelta = params.subarray(2 * K, 3 * K);

  // Update kappa (cumulative)
  const kappa = new Float32Array(K);
  for (let k = 0; k < K; k++) {
    kappa[k] = (kappaPrev[k] as number) + (kappaDelta[k] as number) / 25.0;
  }

  // Clip beta to minimum 0.01
  for (let k = 0; k < K; k++) {
    if ((beta[k] as number) < 0.01) beta[k] = 0.01;
  }

  // Compute phi for each character position, then the window vector
  const window = new Float32Array(alphabetSize);

  for (let u = 0; u < charLen; u++) {
    // phi(u) = sum_k(alpha_k * exp(-(kappa_k - u)^2 / beta_k))
    let phi = 0;
    for (let k = 0; k < K; k++) {
      const diff = (kappa[k] as number) - u;
      phi += (alpha[k] as number) * Math.exp(-(diff * diff) / (beta[k] as number));
    }

    // window += phi * oneHot(char_u)
    const charOffset = u * alphabetSize;
    for (let j = 0; j < alphabetSize; j++) {
      window[j] = (window[j] ?? 0) + phi * (charOneHots[charOffset + j] as number);
    }
  }

  return [window, kappa];
}
