/** Matrix-vector multiply: mat is row-major [rows x cols], vec is [cols]. Returns [rows]. */
export declare function matVecMul(mat: Float32Array, vec: Float32Array, rows: number, cols: number): Float32Array;
/** Element-wise addition. Returns new array. */
export declare function vecAdd(a: Float32Array, b: Float32Array): Float32Array;
/** Element-wise multiply. Returns new array. */
export declare function vecMul(a: Float32Array, b: Float32Array): Float32Array;
/** Element-wise sigmoid. Returns new array. */
export declare function sigmoid(x: Float32Array): Float32Array;
/** Element-wise tanh. Returns new array. */
export declare function tanhVec(x: Float32Array): Float32Array;
/** Element-wise softplus: log(1 + exp(x)). Returns new array. */
export declare function softplus(x: Float32Array): Float32Array;
/** Element-wise exp. Returns new array. */
export declare function expVec(x: Float32Array): Float32Array;
/** Numerically stable softmax. Returns new array. */
export declare function softmax(x: Float32Array): Float32Array;
/** Concatenate multiple Float32Arrays into one. */
export declare function concat(...arrays: Float32Array[]): Float32Array;
/** Add a scalar to every element. Returns new array. */
export declare function vecAddScalar(x: Float32Array, s: number): Float32Array;
/** Return a slice of the array as a new Float32Array. */
export declare function slice(x: Float32Array, start: number, end: number): Float32Array;
//# sourceMappingURL=math.d.ts.map