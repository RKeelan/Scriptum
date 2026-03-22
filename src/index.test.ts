import { describe, expect, test } from "bun:test";

describe("scriptum", () => {
  test("module exports", async () => {
    const mod = await import("./index.js");
    expect(mod).toBeDefined();
  });
});
