import { describe, expect, test } from "bun:test";
import { ALPHABET, ALPHABET_SIZE, encodeText, oneHot } from "./encoding.js";

describe("ALPHABET", () => {
  test("has 73 characters", () => {
    expect(ALPHABET_SIZE).toBe(73);
    expect(ALPHABET.length).toBe(73);
  });

  test("starts with null byte (end-of-sequence)", () => {
    expect(ALPHABET[0]).toBe("\x00");
  });

  test("contains expected characters", () => {
    expect(ALPHABET.includes(" ")).toBe(true);
    expect(ALPHABET.includes("a")).toBe(true);
    expect(ALPHABET.includes("z")).toBe(true);
    expect(ALPHABET.includes("A")).toBe(true);
    expect(ALPHABET.includes("0")).toBe(true);
    expect(ALPHABET.includes("9")).toBe(true);
  });

  test("does not contain Q, X, Z (uppercase)", () => {
    expect(ALPHABET.includes("Q")).toBe(false);
    expect(ALPHABET.includes("X")).toBe(false);
    expect(ALPHABET.includes("Z")).toBe(false);
  });
});

describe("encodeText", () => {
  test("encodes simple text with trailing EOS", () => {
    const encoded = encodeText("ab");
    // 'a' is at index 47, 'b' at 48, then EOS (0)
    expect(encoded).toEqual([47, 48, 0]);
  });

  test("unknown characters map to 0", () => {
    const encoded = encodeText("@");
    // '@' not in alphabet → 0, then trailing EOS → 0
    expect(encoded).toEqual([0, 0]);
  });

  test("empty string returns just EOS", () => {
    expect(encodeText("")).toEqual([0]);
  });

  test("space is encoded correctly", () => {
    const encoded = encodeText(" ");
    expect(encoded[0]).toBe(1); // space is at index 1
  });
});

describe("oneHot", () => {
  test("creates correct one-hot vector", () => {
    const vec = oneHot(2, 5);
    expect(vec.length).toBe(5);
    expect(vec[0]).toBe(0);
    expect(vec[1]).toBe(0);
    expect(vec[2]).toBe(1);
    expect(vec[3]).toBe(0);
    expect(vec[4]).toBe(0);
  });

  test("out of bounds index produces zero vector", () => {
    const vec = oneHot(10, 5);
    for (let i = 0; i < 5; i++) {
      expect(vec[i]).toBe(0);
    }
  });
});
