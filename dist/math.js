/** Matrix-vector multiply: mat is row-major [rows x cols], vec is [cols]. Returns [rows]. */
export function matVecMul(mat, vec, rows, cols) {
    const out = new Float32Array(rows);
    for (let r = 0; r < rows; r++) {
        let sum = 0;
        const offset = r * cols;
        for (let c = 0; c < cols; c++) {
            sum += mat[offset + c] * vec[c];
        }
        out[r] = sum;
    }
    return out;
}
/** Element-wise addition. Returns new array. */
export function vecAdd(a, b) {
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
        out[i] = a[i] + b[i];
    }
    return out;
}
/** Element-wise multiply. Returns new array. */
export function vecMul(a, b) {
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
        out[i] = a[i] * b[i];
    }
    return out;
}
/** Element-wise sigmoid. Returns new array. */
export function sigmoid(x) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
        out[i] = 1 / (1 + Math.exp(-x[i]));
    }
    return out;
}
/** Element-wise tanh. Returns new array. */
export function tanhVec(x) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
        out[i] = Math.tanh(x[i]);
    }
    return out;
}
/** Element-wise softplus: log(1 + exp(x)). Returns new array. */
export function softplus(x) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
        const v = x[i];
        // Numerically stable: for large x, softplus(x) ≈ x
        out[i] = v > 20 ? v : Math.log(1 + Math.exp(v));
    }
    return out;
}
/** Element-wise exp. Returns new array. */
export function expVec(x) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
        out[i] = Math.exp(x[i]);
    }
    return out;
}
/** Numerically stable softmax. Returns new array. */
export function softmax(x) {
    let max = -Infinity;
    for (let i = 0; i < x.length; i++) {
        if (x[i] > max)
            max = x[i];
    }
    const out = new Float32Array(x.length);
    let sum = 0;
    for (let i = 0; i < x.length; i++) {
        const v = Math.exp(x[i] - max);
        out[i] = v;
        sum += v;
    }
    for (let i = 0; i < x.length; i++) {
        out[i] = out[i] / sum;
    }
    return out;
}
/** Concatenate multiple Float32Arrays into one. */
export function concat(...arrays) {
    let totalLen = 0;
    for (const a of arrays)
        totalLen += a.length;
    const out = new Float32Array(totalLen);
    let offset = 0;
    for (const a of arrays) {
        out.set(a, offset);
        offset += a.length;
    }
    return out;
}
/** Add a scalar to every element. Returns new array. */
export function vecAddScalar(x, s) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
        out[i] = x[i] + s;
    }
    return out;
}
/** Return a slice of the array as a new Float32Array. */
export function slice(x, start, end) {
    return x.slice(start, end);
}
//# sourceMappingURL=math.js.map