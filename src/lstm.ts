import { concat, matVecMul, sigmoid, tanhVec, vecAdd, vecAddScalar, vecMul } from "./math.js";

/**
 * Single LSTM step following TensorFlow's LSTMCell conventions.
 *
 * Gate order: i (input), j (new cell content), f (forget), o (output).
 * Kernel shape: [inputSize + numUnits, 4 * numUnits] (row-major).
 * A forget bias of 1.0 is added to the forget gate.
 *
 * @returns [hNew, cNew] — new hidden state and cell state.
 */
export function lstmStep(
  x: Float32Array,
  hPrev: Float32Array,
  cPrev: Float32Array,
  kernel: Float32Array,
  bias: Float32Array,
  numUnits: number,
): [Float32Array, Float32Array] {
  const input = concat(x, hPrev);
  const inputSize = input.length;
  const gateSize = 4 * numUnits;

  // gates = kernel^T * input + bias
  // kernel is [inputSize, gateSize] row-major
  const gates = vecAdd(matVecMul(kernel, input, gateSize, inputSize), bias);

  // Split into four gates
  const iGate = sigmoid(gates.slice(0, numUnits));
  const jGate = tanhVec(gates.slice(numUnits, 2 * numUnits));
  const fGate = sigmoid(vecAddScalar(gates.slice(2 * numUnits, 3 * numUnits), 1.0));
  const oGate = sigmoid(gates.slice(3 * numUnits, 4 * numUnits));

  // c_new = f * c_prev + i * j
  const cNew = vecAdd(vecMul(fGate, cPrev), vecMul(iGate, jGate));
  // h_new = o * tanh(c_new)
  const hNew = vecMul(oGate, tanhVec(cNew));

  return [hNew, cNew];
}
