import { describe, expect, test } from "bun:test";
import { loadWeights } from "./weights.js";

/** Build a minimal binary weight file with one tensor (float32). */
function buildTestBuffer(name: string, shape: number[], float32Data: number[]): ArrayBuffer {
  const nameBytes = new TextEncoder().encode(name);
  // Header: 4 (magic) + 4 (count) = 8
  // Tensor: 2 (nameLen) + nameBytes.length + 1 (ndims) + shape.length*4 + float32Data.length*4
  const size = 8 + 2 + nameBytes.length + 1 + shape.length * 4 + float32Data.length * 4;
  const buffer = new ArrayBuffer(size);
  const view = new DataView(buffer);
  let offset = 0;

  // Header
  view.setUint32(offset, 0x50524353, true);
  offset += 4; // "SCRP"
  view.setUint32(offset, 1, true);
  offset += 4; // tensor count

  // Tensor
  view.setUint16(offset, nameBytes.length, true);
  offset += 2;
  new Uint8Array(buffer, offset, nameBytes.length).set(nameBytes);
  offset += nameBytes.length;
  view.setUint8(offset, shape.length);
  offset += 1;
  for (const dim of shape) {
    view.setUint32(offset, dim, true);
    offset += 4;
  }
  for (const val of float32Data) {
    view.setFloat32(offset, val, true);
    offset += 4;
  }

  return buffer;
}

describe("loadWeights", () => {
  test("parses a single 1D tensor", () => {
    const buf = buildTestBuffer("test_bias", [3], [1.5, -2.0, 0.0]);
    const weights = loadWeights(buf);

    const tensor = weights.test_bias;
    if (!tensor) throw new Error("missing test_bias");
    expect(tensor.shape).toEqual([3]);
    expect(tensor.data.length).toBe(3);
    expect(tensor.data[0]).toBeCloseTo(1.5, 4);
    expect(tensor.data[1]).toBeCloseTo(-2.0, 4);
    expect(tensor.data[2]).toBeCloseTo(0, 4);
  });

  test("parses a 2D tensor", () => {
    const buf = buildTestBuffer("test_kernel", [2, 2], [1.0, 2.0, 3.0, 4.0]);
    const weights = loadWeights(buf);

    const tensor = weights.test_kernel;
    if (!tensor) throw new Error("missing test_kernel");
    expect(tensor.shape).toEqual([2, 2]);
    expect(tensor.data.length).toBe(4);
    expect(tensor.data[0]).toBeCloseTo(1.0, 4);
    expect(tensor.data[1]).toBeCloseTo(2.0, 4);
    expect(tensor.data[2]).toBeCloseTo(3.0, 4);
    expect(tensor.data[3]).toBeCloseTo(4.0, 4);
  });

  test("rejects invalid magic", () => {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setUint32(0, 0xdeadbeef, true);
    expect(() => loadWeights(buf)).toThrow("Invalid weight file");
  });
});
