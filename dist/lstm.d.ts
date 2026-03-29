/**
 * Single LSTM step following TensorFlow's LSTMCell conventions.
 *
 * Gate order: i (input), j (new cell content), f (forget), o (output).
 * Kernel shape: [inputSize + numUnits, 4 * numUnits] (row-major).
 * A forget bias of 1.0 is added to the forget gate.
 *
 * @returns [hNew, cNew] — new hidden state and cell state.
 */
export declare function lstmStep(x: Float32Array, hPrev: Float32Array, cPrev: Float32Array, kernel: Float32Array, bias: Float32Array, numUnits: number): [Float32Array, Float32Array];
//# sourceMappingURL=lstm.d.ts.map