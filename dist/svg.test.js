import { describe, expect, test } from "bun:test";
import { outlinePath, strokesToCoords, strokesToOutlinePaths, strokesToSvgPaths } from "./svg.js";
// Helper: create stroke points from simple coordinate pairs with optional penUp
function makePoints(coords) {
    return coords.map(([dx, dy, penUp]) => ({ dx, dy, penUp: penUp ?? false }));
}
describe("strokesToCoords", () => {
    test("returns empty array for empty input", () => {
        expect(strokesToCoords([])).toEqual([]);
    });
    test("converts offsets to absolute coordinates", () => {
        const points = makePoints([
            [1, 0],
            [1, 0],
            [1, 0, true],
        ]);
        const strokes = strokesToCoords(points);
        expect(strokes).toHaveLength(1);
        expect(strokes[0]).toHaveLength(3);
        // x should be cumulative (1, 2, 3); y stays 0 (negated = 0)
        // Alignment may shift values slightly, but ordering should be preserved
        const first = strokes[0];
        for (let i = 1; i < first.length; i++) {
            expect(first[i].x).toBeGreaterThan(first[i - 1].x);
        }
    });
    test("splits at pen-up boundaries", () => {
        const points = makePoints([
            [1, 0, true], // stroke 1 (single point then penUp)
            [1, 0],
            [1, 0, true], // stroke 2
        ]);
        const strokes = strokesToCoords(points);
        expect(strokes).toHaveLength(2);
    });
    test("returns same stroke count as strokesToSvgPaths", () => {
        const points = makePoints([
            [1, 0],
            [2, 1, true],
            [1, -1],
            [1, 0, true],
        ]);
        const coords = strokesToCoords(points);
        const paths = strokesToSvgPaths(points);
        expect(coords).toHaveLength(paths.length);
    });
});
describe("outlinePath", () => {
    test("returns empty string for empty input", () => {
        expect(outlinePath([])).toBe("");
    });
    test("handles single point (returns a circle arc)", () => {
        const d = outlinePath([{ x: 5, y: 5 }]);
        expect(d).toContain("M");
        expect(d).toContain("A");
        expect(d).toContain("Z");
    });
    test("handles two points (returns closed path)", () => {
        const d = outlinePath([
            { x: 0, y: 0 },
            { x: 10, y: 0 },
        ]);
        expect(d).toContain("M");
        expect(d).toContain("Z");
    });
    test("produces a closed path for normal strokes", () => {
        const stroke = [
            { x: 0, y: 0 },
            { x: 2, y: 1 },
            { x: 4, y: 0 },
            { x: 6, y: -1 },
            { x: 8, y: 0 },
        ];
        const d = outlinePath(stroke);
        expect(d.startsWith("M")).toBe(true);
        expect(d.endsWith("Z")).toBe(true);
    });
    test("respects custom options", () => {
        const stroke = [
            { x: 0, y: 0 },
            { x: 5, y: 0 },
            { x: 10, y: 0 },
        ];
        const thin = outlinePath(stroke, { minWidth: 0.05, maxWidth: 0.1 });
        const thick = outlinePath(stroke, { minWidth: 1, maxWidth: 3 });
        // Thick path should differ from thin
        expect(thin).not.toBe(thick);
    });
});
describe("strokesToOutlinePaths", () => {
    test("returns same stroke count as strokesToSvgPaths", () => {
        const points = makePoints([
            [1, 0],
            [2, 1],
            [1, 0, true],
            [1, -1],
            [2, 0, true],
        ]);
        const outlines = strokesToOutlinePaths(points);
        const paths = strokesToSvgPaths(points);
        expect(outlines).toHaveLength(paths.length);
    });
    test("all paths are closed", () => {
        const points = makePoints([
            [1, 0],
            [2, 1],
            [1, 0, true],
        ]);
        const outlines = strokesToOutlinePaths(points);
        for (const d of outlines) {
            expect(d.endsWith("Z")).toBe(true);
        }
    });
});
//# sourceMappingURL=svg.test.js.map