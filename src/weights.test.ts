import { describe, expect, test } from "bun:test";
import { loadWeights } from "./weights.js";

/** Build a minimal binary weight file with one tensor. */
function buildTestBuffer(
  name: string,
  shape: number[],
  scale: number,
  int8Data: number[],
): ArrayBuffer {
  const nameBytes = new TextEncoder().encode(name);
  // Header: 4 (magic) + 4 (version) + 4 (count) = 12
  // Tensor: 2 (nameLen) + nameBytes.length + 1 (ndims) + shape.length*4 + 4 (scale) + int8Data.length
  const size = 12 + 2 + nameBytes.length + 1 + shape.length * 4 + 4 + int8Data.length;
  const buffer = new ArrayBuffer(size);
  const view = new DataView(buffer);
  let offset = 0;

  // Header
  view.setUint32(offset, 0x50524353, true);
  offset += 4; // "SCRP"
  view.setUint32(offset, 1, true);
  offset += 4; // version
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
  view.setFloat32(offset, scale, true);
  offset += 4;
  for (const val of int8Data) {
    view.setInt8(offset, val);
    offset += 1;
  }

  return buffer;
}

describe("loadWeights", () => {
  test("parses a single 1D tensor", () => {
    // scale=1.27, int8=[127, -127, 0] → float32=[1.27, -1.27, 0]
    const buf = buildTestBuffer("test_bias", [3], 1.27, [127, -127, 0]);
    const weights = loadWeights(buf);

    const tensor = weights.test_bias;
    if (!tensor) throw new Error("missing test_bias");
    expect(tensor.shape).toEqual([3]);
    expect(tensor.data.length).toBe(3);
    expect(tensor.data[0]).toBeCloseTo(1.27, 4);
    expect(tensor.data[1]).toBeCloseTo(-1.27, 4);
    expect(tensor.data[2]).toBeCloseTo(0, 4);
  });

  test("parses a 2D tensor", () => {
    // 2x2 matrix, scale=2.54
    const buf = buildTestBuffer("test_kernel", [2, 2], 2.54, [127, 0, -64, 64]);
    const weights = loadWeights(buf);

    const tensor = weights.test_kernel;
    if (!tensor) throw new Error("missing test_kernel");
    expect(tensor.shape).toEqual([2, 2]);
    expect(tensor.data.length).toBe(4);
    expect(tensor.data[0]).toBeCloseTo(2.54, 4);
    expect(tensor.data[1]).toBeCloseTo(0, 4);
    expect(tensor.data[2]).toBeCloseTo((-2.54 * 64) / 127, 3);
    expect(tensor.data[3]).toBeCloseTo((2.54 * 64) / 127, 3);
  });

  test("rejects invalid magic", () => {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setUint32(0, 0xdeadbeef, true);
    expect(() => loadWeights(buf)).toThrow("Invalid weight file");
  });

  test("rejects unsupported version", () => {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setUint32(0, 0x50524353, true);
    view.setUint32(4, 99, true);
    expect(() => loadWeights(buf)).toThrow("Unsupported weight file version");
  });
});
