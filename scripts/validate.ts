/**
 * Compare TypeScript forward pass against Python reference values.
 * Run with: bun scripts/validate.ts
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { attentionStep } from "../src/attention.js";
import { ALPHABET_SIZE, encodeText, oneHot } from "../src/encoding.js";
import { lstmStep } from "../src/lstm.js";
import { concat, matVecMul, vecAdd } from "../src/math.js";
import { parseMdnOutput } from "../src/mdn.js";
import { loadWeights } from "../src/weights.js";

const NUM_UNITS = 400;
const NUM_ATTENTION_COMPONENTS = 10;
const NUM_MIXTURE_COMPONENTS = 20;

interface StepData {
  step: number;
  input: number[];
  h1: number[];
  c1: number[];
  h2: number[];
  h3: number[];
  kappa: number[];
  window: number[];
  gmm_raw: number[];
}

interface ReferenceData {
  text: string;
  char_indices: number[];
  bias: number;
  num_steps: number;
  steps: StepData[];
}

function maxAbsError(a: Float32Array | number[], b: number[]): number {
  let max = 0;
  for (let i = 0; i < b.length; i++) {
    const ai = a instanceof Float32Array ? (a[i] ?? 0) : (a[i] ?? 0);
    const diff = Math.abs(ai - (b[i] ?? 0));
    if (diff > max) max = diff;
  }
  return max;
}

function meanAbsError(a: Float32Array | number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < b.length; i++) {
    const ai = a instanceof Float32Array ? (a[i] ?? 0) : (a[i] ?? 0);
    sum += Math.abs(ai - (b[i] ?? 0));
  }
  return sum / b.length;
}

async function main() {
  // Load reference values
  const refPath = resolve(import.meta.dir, "reference-values.json");
  const refData: ReferenceData = JSON.parse(readFileSync(refPath, "utf-8"));

  console.log(`Reference: text='${refData.text}', ${refData.num_steps} steps, bias=${refData.bias}`);

  // Load weights
  const weightsPath = resolve(import.meta.dir, "..", "weights", "synthesis.bin");
  const weightsBuffer = readFileSync(weightsPath);
  const weights = loadWeights(weightsBuffer.buffer.slice(
    weightsBuffer.byteOffset,
    weightsBuffer.byteOffset + weightsBuffer.byteLength,
  ));

  const lstm1Kernel = weights.lstm1_kernel?.data;
  const lstm1Bias = weights.lstm1_bias?.data;
  const lstm2Kernel = weights.lstm2_kernel?.data;
  const lstm2Bias = weights.lstm2_bias?.data;
  const lstm3Kernel = weights.lstm3_kernel?.data;
  const lstm3Bias = weights.lstm3_bias?.data;
  const attnWeights = weights.attention_weights?.data;
  const attnBias = weights.attention_bias?.data;
  const gmmWeights = weights.gmm_weights?.data;
  const gmmBias = weights.gmm_bias?.data;

  if (!lstm1Kernel || !lstm1Bias || !lstm2Kernel || !lstm2Bias ||
      !lstm3Kernel || !lstm3Bias || !attnWeights || !attnBias ||
      !gmmWeights || !gmmBias) {
    throw new Error("Missing weight tensors");
  }

  // Encode text
  const charIndices = encodeText(refData.text);
  console.log(`TS char indices: [${charIndices.join(", ")}]`);
  console.log(`Ref char indices: [${refData.char_indices.join(", ")}]`);

  const charLen = charIndices.length;
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

  let allOk = true;

  for (const refStep of refData.steps) {
    const x = new Float32Array(refStep.input);

    // LSTM 1
    const s1In = concat(window, x);
    [h1, c1] = lstmStep(s1In, h1, c1, lstm1Kernel, lstm1Bias, NUM_UNITS);

    // Attention
    [window, kappa] = attentionStep(
      window, x, h1, kappa, charOneHots, charLen, ALPHABET_SIZE,
      attnWeights, attnBias, NUM_ATTENTION_COMPONENTS,
    );

    // LSTM 2
    const s2In = concat(x, h1, window);
    [h2, c2] = lstmStep(s2In, h2, c2, lstm2Kernel, lstm2Bias, NUM_UNITS);

    // LSTM 3
    const s3In = concat(x, h2, window);
    [h3, c3] = lstmStep(s3In, h3, c3, lstm3Kernel, lstm3Bias, NUM_UNITS);

    // GMM
    const rawOutput = vecAdd(
      matVecMul(gmmWeights, h3, NUM_MIXTURE_COMPONENTS * 6 + 1, NUM_UNITS),
      gmmBias,
    );

    // Compare
    const h1Err = maxAbsError(h1, refStep.h1);
    const windowErr = maxAbsError(window, refStep.window);
    const kappaErr = maxAbsError(kappa, refStep.kappa);
    const h3Err = maxAbsError(h3, refStep.h3);
    const gmmErr = maxAbsError(rawOutput, refStep.gmm_raw);

    const h1Mean = meanAbsError(h1, refStep.h1);
    const gmmMean = meanAbsError(rawOutput, refStep.gmm_raw);

    const ok = h1Err < 0.1 && windowErr < 0.5 && kappaErr < 0.01 && h3Err < 0.5;

    console.log(
      `Step ${refStep.step}: ` +
      `h1 max=${h1Err.toFixed(6)} mean=${h1Mean.toFixed(6)} | ` +
      `window max=${windowErr.toFixed(6)} | ` +
      `kappa max=${kappaErr.toFixed(6)} | ` +
      `h3 max=${h3Err.toFixed(6)} | ` +
      `gmm max=${gmmErr.toFixed(6)} mean=${gmmMean.toFixed(6)} ` +
      (ok ? "OK" : "FAIL"),
    );

    if (!ok) allOk = false;
  }

  console.log(allOk ? "\nAll steps within tolerance." : "\nSome steps exceeded tolerance.");
  process.exit(allOk ? 0 : 1);
}

main();
