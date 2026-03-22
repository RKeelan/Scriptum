/** The 73-character alphabet matching the IAM training data. Index 0 is end-of-sequence. */
export const ALPHABET: readonly string[] = [
  "\x00",
  " ",
  "!",
  '"',
  "#",
  "'",
  "(",
  ")",
  ",",
  "-",
  ".",
  "0",
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  ":",
  ";",
  "?",
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "Y",
  "a",
  "b",
  "c",
  "d",
  "e",
  "f",
  "g",
  "h",
  "i",
  "j",
  "k",
  "l",
  "m",
  "n",
  "o",
  "p",
  "q",
  "r",
  "s",
  "t",
  "u",
  "v",
  "w",
  "x",
  "y",
  "z",
] as const;

export const ALPHABET_SIZE = ALPHABET.length; // 73

/** Map from character to index. Unknown characters map to 0 (end-of-sequence). */
const charToIndex = new Map<string, number>();
for (let i = 0; i < ALPHABET.length; i++) {
  charToIndex.set(ALPHABET[i] as string, i);
}

/** Encode a string as an array of character indices, with a trailing end-of-sequence (0). */
export function encodeText(text: string): number[] {
  const indices: number[] = [];
  for (const ch of text) {
    indices.push(charToIndex.get(ch) ?? 0);
  }
  indices.push(0); // end-of-sequence
  return indices;
}

/** Create a one-hot Float32Array of the given size with a 1 at the given index. */
export function oneHot(index: number, size: number): Float32Array {
  const vec = new Float32Array(size);
  if (index >= 0 && index < size) {
    vec[index] = 1;
  }
  return vec;
}
