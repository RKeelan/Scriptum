import { describe, expect, test } from "bun:test";
import { concat, expVec, matVecMul, sigmoid, slice, softmax, softplus, tanhVec, vecAdd, vecAddScalar, vecMul, } from "./math.js";
function approxEqual(a, b) {
    expect(a.length).toBe(b.length);
    for (let i = 0; i < a.length; i++) {
        expect(a[i]).toBeCloseTo(b[i], 5);
    }
}
describe("matVecMul", () => {
    test("2x3 matrix times 3-vector", () => {
        // [[1,2,3],[4,5,6]] * [1,0,1] = [4, 10]
        const mat = new Float32Array([1, 2, 3, 4, 5, 6]);
        const vec = new Float32Array([1, 0, 1]);
        const result = matVecMul(mat, vec, 2, 3);
        approxEqual(result, new Float32Array([4, 10]));
    });
    test("identity matrix", () => {
        const mat = new Float32Array([1, 0, 0, 1]);
        const vec = new Float32Array([3, 7]);
        approxEqual(matVecMul(mat, vec, 2, 2), new Float32Array([3, 7]));
    });
});
describe("vecAdd", () => {
    test("adds element-wise", () => {
        const a = new Float32Array([1, 2, 3]);
        const b = new Float32Array([4, 5, 6]);
        approxEqual(vecAdd(a, b), new Float32Array([5, 7, 9]));
    });
});
describe("vecMul", () => {
    test("multiplies element-wise", () => {
        const a = new Float32Array([2, 3, 4]);
        const b = new Float32Array([5, 6, 7]);
        approxEqual(vecMul(a, b), new Float32Array([10, 18, 28]));
    });
});
describe("sigmoid", () => {
    test("known values", () => {
        const x = new Float32Array([0, 1, -1, 10, -10]);
        const result = sigmoid(x);
        expect(result[0]).toBeCloseTo(0.5, 5);
        expect(result[1]).toBeCloseTo(0.7310586, 5);
        expect(result[2]).toBeCloseTo(0.2689414, 5);
        expect(result[3]).toBeCloseTo(1.0, 4);
        expect(result[4]).toBeCloseTo(0.0, 4);
    });
});
describe("tanhVec", () => {
    test("known values", () => {
        const x = new Float32Array([0, 1, -1]);
        const result = tanhVec(x);
        expect(result[0]).toBeCloseTo(0, 5);
        expect(result[1]).toBeCloseTo(0.7615942, 5);
        expect(result[2]).toBeCloseTo(-0.7615942, 5);
    });
});
describe("softplus", () => {
    test("known values", () => {
        const x = new Float32Array([0, 1, -1, 25]);
        const result = softplus(x);
        expect(result[0]).toBeCloseTo(Math.log(2), 5);
        expect(result[1]).toBeCloseTo(Math.log(1 + Math.E), 5);
        expect(result[2]).toBeCloseTo(Math.log(1 + 1 / Math.E), 5);
        expect(result[3]).toBeCloseTo(25, 4); // large x fallback
    });
});
describe("expVec", () => {
    test("known values", () => {
        const x = new Float32Array([0, 1, -1]);
        const result = expVec(x);
        expect(result[0]).toBeCloseTo(1, 5);
        expect(result[1]).toBeCloseTo(Math.E, 5);
        expect(result[2]).toBeCloseTo(1 / Math.E, 5);
    });
});
describe("softmax", () => {
    test("uniform input gives uniform output", () => {
        const x = new Float32Array([1, 1, 1]);
        const result = softmax(x);
        for (let i = 0; i < 3; i++) {
            expect(result[i]).toBeCloseTo(1 / 3, 5);
        }
    });
    test("sums to 1", () => {
        const x = new Float32Array([1, 2, 3, 4]);
        const result = softmax(x);
        let sum = 0;
        for (let i = 0; i < result.length; i++)
            sum += result[i];
        expect(sum).toBeCloseTo(1, 5);
    });
    test("largest input gets largest probability", () => {
        const x = new Float32Array([1, 5, 2]);
        const result = softmax(x);
        expect(result[1]).toBeGreaterThan(result[0]);
        expect(result[1]).toBeGreaterThan(result[2]);
    });
});
describe("concat", () => {
    test("concatenates multiple arrays", () => {
        const a = new Float32Array([1, 2]);
        const b = new Float32Array([3]);
        const c = new Float32Array([4, 5, 6]);
        approxEqual(concat(a, b, c), new Float32Array([1, 2, 3, 4, 5, 6]));
    });
    test("handles empty arrays", () => {
        const a = new Float32Array([1, 2]);
        const b = new Float32Array(0);
        approxEqual(concat(a, b), new Float32Array([1, 2]));
    });
});
describe("vecAddScalar", () => {
    test("adds scalar to each element", () => {
        const x = new Float32Array([1, 2, 3]);
        approxEqual(vecAddScalar(x, 10), new Float32Array([11, 12, 13]));
    });
});
describe("slice", () => {
    test("slices correctly", () => {
        const x = new Float32Array([10, 20, 30, 40, 50]);
        approxEqual(slice(x, 1, 4), new Float32Array([20, 30, 40]));
    });
});
//# sourceMappingURL=math.test.js.map