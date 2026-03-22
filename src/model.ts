import { attentionStep } from "./attention.js";
import { ALPHABET_SIZE, encodeText, oneHot } from "./encoding.js";
import { lstmStep } from "./lstm.js";
import { concat, matVecMul, vecAdd } from "./math.js";
import { parseMdnOutput, sampleFromMixture } from "./mdn.js";
import type { StrokePoint, WeightDict, WeightTensor } from "./types.js";

const NUM_UNITS = 400;
const NUM_ATTENTION_COMPONENTS = 10;
const NUM_MIXTURE_COMPONENTS = 20;

function getWeight(weights: WeightDict, name: string): WeightTensor {
  const w = weights[name];
  if (!w) throw new Error(`Missing weight tensor: ${name}`);
  return w;
}

export class SynthesisModel {
  private readonly lstm1Kernel: Float32Array;
  private readonly lstm1Bias: Float32Array;
  private readonly lstm2Kernel: Float32Array;
  private readonly lstm2Bias: Float32Array;
  private readonly lstm3Kernel: Float32Array;
  private readonly lstm3Bias: Float32Array;
  private readonly attnWeights: Float32Array;
  private readonly attnBias: Float32Array;
  private readonly gmmWeights: Float32Array;
  private readonly gmmBias: Float32Array;

  constructor(weights: WeightDict) {
    this.lstm1Kernel = getWeight(weights, "lstm1_kernel").data;
    this.lstm1Bias = getWeight(weights, "lstm1_bias").data;
    this.lstm2Kernel = getWeight(weights, "lstm2_kernel").data;
    this.lstm2Bias = getWeight(weights, "lstm2_bias").data;
    this.lstm3Kernel = getWeight(weights, "lstm3_kernel").data;
    this.lstm3Bias = getWeight(weights, "lstm3_bias").data;
    this.attnWeights = getWeight(weights, "attention_weights").data;
    this.attnBias = getWeight(weights, "attention_bias").data;
    this.gmmWeights = getWeight(weights, "gmm_weights").data;
    this.gmmBias = getWeight(weights, "gmm_bias").data;
  }

  /**
   * Generate handwriting strokes for the given text.
   *
   * @param text - Text to synthesize
   * @param options - Generation options
   * @returns Array of stroke points (dx, dy, penUp)
   */
  generate(text: string, options?: { bias?: number; maxSteps?: number }): StrokePoint[] {
    const bias = options?.bias ?? 0.5;
    const maxSteps = options?.maxSteps ?? 40 * text.length;

    // Encode characters
    const charIndices = encodeText(text);
    const charLen = charIndices.length;

    // Build flat one-hot character matrix [charLen * ALPHABET_SIZE]
    const charOneHots = new Float32Array(charLen * ALPHABET_SIZE);
    for (let i = 0; i < charLen; i++) {
      const oh = oneHot(charIndices[i] as number, ALPHABET_SIZE);
      charOneHots.set(oh, i * ALPHABET_SIZE);
    }

    // Initialise states
    let h1 = new Float32Array(NUM_UNITS) as Float32Array;
    let c1 = new Float32Array(NUM_UNITS) as Float32Array;
    let h2 = new Float32Array(NUM_UNITS) as Float32Array;
    let c2 = new Float32Array(NUM_UNITS) as Float32Array;
    let h3 = new Float32Array(NUM_UNITS) as Float32Array;
    let c3 = new Float32Array(NUM_UNITS) as Float32Array;
    let kappa = new Float32Array(NUM_ATTENTION_COMPONENTS) as Float32Array;
    let window = new Float32Array(ALPHABET_SIZE) as Float32Array;

    // Initial input: pen up
    let x = new Float32Array(3);
    x[2] = 1; // pen up

    const points: StrokePoint[] = [];

    for (let step = 0; step < maxSteps; step++) {
      // LSTM 1: input = concat(window, x)
      const s1In = concat(window, x);
      [h1, c1] = lstmStep(s1In, h1, c1, this.lstm1Kernel, this.lstm1Bias, NUM_UNITS);

      // Attention: updates window and kappa
      [window, kappa] = attentionStep(
        window,
        x,
        h1,
        kappa,
        charOneHots,
        charLen,
        ALPHABET_SIZE,
        this.attnWeights,
        this.attnBias,
        NUM_ATTENTION_COMPONENTS,
      );

      // LSTM 2: input = concat(x, h1, window)
      const s2In = concat(x, h1, window);
      [h2, c2] = lstmStep(s2In, h2, c2, this.lstm2Kernel, this.lstm2Bias, NUM_UNITS);

      // LSTM 3: input = concat(x, h2, window)
      const s3In = concat(x, h2, window);
      [h3, c3] = lstmStep(s3In, h3, c3, this.lstm3Kernel, this.lstm3Bias, NUM_UNITS);

      // GMM output from h3
      const rawOutput = vecAdd(
        matVecMul(this.gmmWeights, h3, NUM_MIXTURE_COMPONENTS * 6 + 1, NUM_UNITS),
        this.gmmBias,
      );

      // Parse and sample
      const mdnParams = parseMdnOutput(rawOutput, NUM_MIXTURE_COMPONENTS, bias);
      const [dx, dy, penUp] = sampleFromMixture(mdnParams);

      points.push({ dx, dy, penUp });

      // Check termination: attention past last character
      let maxPhiIdx = 0;
      let maxPhiVal = -Infinity;
      // Recompute phi for termination check (approximate: use kappa)
      for (let u = 0; u < charLen; u++) {
        let phi = 0;
        for (let k = 0; k < NUM_ATTENTION_COMPONENTS; k++) {
          const diff = (kappa[k] as number) - u;
          // Use a simple approximation — if kappa is past the last char, stop
          phi += Math.exp(-diff * diff);
        }
        if (phi > maxPhiVal) {
          maxPhiVal = phi;
          maxPhiIdx = u;
        }
      }

      const pastFinalChar = maxPhiIdx >= charLen - 1;
      if (pastFinalChar && penUp) break;

      // Next input
      x = new Float32Array([dx, dy, penUp ? 1 : 0]);
    }

    return points;
  }
}
