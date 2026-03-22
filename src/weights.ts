import type { WeightDict } from "./types.js";

const MAGIC = 0x50524353; // "SCRP" as little-endian uint32
const SUPPORTED_VERSION = 1;

/**
 * Parse a binary weight file into a WeightDict.
 * Dequantises int8 values back to float32 using stored scale factors.
 */
export function loadWeights(buffer: ArrayBuffer): WeightDict {
  const view = new DataView(buffer);
  let offset = 0;

  const magic = view.getUint32(offset, true);
  offset += 4;
  if (magic !== MAGIC) {
    throw new Error(`Invalid weight file: expected magic SCRP, got 0x${magic.toString(16)}`);
  }

  const version = view.getUint32(offset, true);
  offset += 4;
  if (version !== SUPPORTED_VERSION) {
    throw new Error(`Unsupported weight file version: ${version}`);
  }

  const tensorCount = view.getUint32(offset, true);
  offset += 4;

  const weights: WeightDict = {};

  for (let t = 0; t < tensorCount; t++) {
    const nameLen = view.getUint16(offset, true);
    offset += 2;

    const nameBytes = new Uint8Array(buffer, offset, nameLen);
    const name = new TextDecoder().decode(nameBytes);
    offset += nameLen;

    const ndims = view.getUint8(offset);
    offset += 1;

    const shape: number[] = [];
    let totalElements = 1;
    for (let d = 0; d < ndims; d++) {
      const dim = view.getUint32(offset, true);
      offset += 4;
      shape.push(dim);
      totalElements *= dim;
    }

    const scale = view.getFloat32(offset, true);
    offset += 4;

    // Dequantise: float32 = int8 / 127 * scale
    const data = new Float32Array(totalElements);
    const factor = scale / 127;
    for (let i = 0; i < totalElements; i++) {
      data[i] = view.getInt8(offset + i) * factor;
    }
    offset += totalElements;

    weights[name] = { data, shape };
  }

  return weights;
}

/** Fetch a weight file from a URL and parse it. */
export async function fetchWeights(url: string): Promise<WeightDict> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch weights from ${url}: ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  return loadWeights(buffer);
}
