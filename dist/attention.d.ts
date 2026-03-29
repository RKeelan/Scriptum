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
export declare function attentionStep(prevWindow: Float32Array, x: Float32Array, h1: Float32Array, kappaPrev: Float32Array, charOneHots: Float32Array, charLen: number, alphabetSize: number, weights: Float32Array, bias: Float32Array, K: number): [Float32Array, Float32Array];
//# sourceMappingURL=attention.d.ts.map