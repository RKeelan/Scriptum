import { describe, expect, test } from "bun:test";
import { lstmStep } from "./lstm.js";

describe("lstmStep", () => {
  test("produces correct output for tiny LSTM", () => {
    // numUnits=2, inputSize=1, so kernel is [4*2, 1+2] = [8, 3] row-major
    // input x=[1], hPrev=[0,0], cPrev=[0,0]
    const numUnits = 2;

    // Build a simple kernel: all zeros except we set specific values
    // kernel[row][col] where rows=8 (4*numUnits), cols=3 (1+numUnits)
    // Gate order: i, j, f, o
    const kernel = new Float32Array(8 * 3);
    // Set all kernel values to 0.5 for simplicity
    kernel.fill(0.5);

    const bias = new Float32Array(8);
    bias.fill(0);

    const x = new Float32Array([1.0]);
    const hPrev = new Float32Array([0.0, 0.0]);
    const cPrev = new Float32Array([0.0, 0.0]);

    const [hNew, cNew] = lstmStep(x, hPrev, cPrev, kernel, bias, numUnits);

    expect(hNew.length).toBe(2);
    expect(cNew.length).toBe(2);

    // With all weights=0.5, input=[1,0,0]:
    // gates = kernel * [1,0,0] + bias = first column of kernel = [0.5]*8
    // i = sigmoid(0.5) ≈ 0.6225
    // j = tanh(0.5) ≈ 0.4621
    // f = sigmoid(0.5 + 1.0) = sigmoid(1.5) ≈ 0.8176 (forget bias)
    // o = sigmoid(0.5) ≈ 0.6225
    // c = f*0 + i*j ≈ 0.6225 * 0.4621 ≈ 0.2877
    // h = o * tanh(c) ≈ 0.6225 * tanh(0.2877) ≈ 0.6225 * 0.2807 ≈ 0.1747
    expect(cNew[0]).toBeCloseTo(0.2877, 3);
    expect(hNew[0]).toBeCloseTo(0.1747, 3);
  });

  test("forget gate preserves cell state", () => {
    // With high forget bias and zero input gate, cell state should be mostly preserved
    const numUnits = 1;
    const kernel = new Float32Array(4); // [4, 1] — just input, no hidden (will concat with h=0)
    kernel.fill(0);

    const bias = new Float32Array(4);
    // i gate bias = -10 (sigmoid → ~0, blocks new input)
    bias[0] = -10;
    // j gate bias = 0
    bias[1] = 0;
    // f gate bias = 0 (+ forget_bias=1.0 → sigmoid(1) ≈ 0.73)
    bias[2] = 0;
    // o gate bias = 10 (sigmoid → ~1, passes output)
    bias[3] = 10;

    const x = new Float32Array(0); // no input features
    const hPrev = new Float32Array([0.0]);
    const cPrev = new Float32Array([5.0]);

    const [hNew, cNew] = lstmStep(x, hPrev, cPrev, kernel, bias, numUnits);

    // f ≈ sigmoid(1.0) ≈ 0.7311, i ≈ 0, so c_new ≈ 0.7311 * 5 ≈ 3.655
    expect(cNew[0]).toBeCloseTo(3.655, 2);
    // h = o * tanh(c_new) ≈ 1.0 * tanh(3.655) ≈ 0.9987
    expect(hNew[0]).toBeCloseTo(Math.tanh(3.655), 3);
  });
});
