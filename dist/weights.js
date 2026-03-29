const MAGIC = 0x50524353; // "SCRP" as little-endian uint32
/**
 * Parse a binary weight file into a WeightDict.
 *
 * Binary format:
 *   Header: magic "SCRP" (4 bytes), tensor_count (uint32 LE)
 *   Per tensor: name_length (uint16 LE), name (utf8), ndims (uint8),
 *               shape (ndims x uint32 LE), data (float32 LE[product(shape)])
 */
export function loadWeights(buffer) {
    const view = new DataView(buffer);
    let offset = 0;
    const magic = view.getUint32(offset, true);
    offset += 4;
    if (magic !== MAGIC) {
        throw new Error(`Invalid weight file: expected magic SCRP, got 0x${magic.toString(16)}`);
    }
    const tensorCount = view.getUint32(offset, true);
    offset += 4;
    const weights = {};
    for (let t = 0; t < tensorCount; t++) {
        const nameLen = view.getUint16(offset, true);
        offset += 2;
        const nameBytes = new Uint8Array(buffer, offset, nameLen);
        const name = new TextDecoder().decode(nameBytes);
        offset += nameLen;
        const ndims = view.getUint8(offset);
        offset += 1;
        const shape = [];
        let totalElements = 1;
        for (let d = 0; d < ndims; d++) {
            const dim = view.getUint32(offset, true);
            offset += 4;
            shape.push(dim);
            totalElements *= dim;
        }
        const data = new Float32Array(totalElements);
        for (let i = 0; i < totalElements; i++) {
            data[i] = view.getFloat32(offset, true);
            offset += 4;
        }
        weights[name] = { data, shape };
    }
    return weights;
}
/** Fetch a weight file from a URL and parse it. */
export async function fetchWeights(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch weights from ${url}: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    return loadWeights(buffer);
}
//# sourceMappingURL=weights.js.map