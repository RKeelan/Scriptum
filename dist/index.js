// src/encoding.ts
var ALPHABET = [
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
  "z"
];
var ALPHABET_SIZE = ALPHABET.length;
var charToIndex = new Map;
for (let i = 0;i < ALPHABET.length; i++) {
  charToIndex.set(ALPHABET[i], i);
}
function encodeText(text) {
  const indices = [];
  for (const ch of text) {
    indices.push(charToIndex.get(ch) ?? 0);
  }
  indices.push(0);
  return indices;
}
function oneHot(index, size) {
  const vec = new Float32Array(size);
  if (index >= 0 && index < size) {
    vec[index] = 1;
  }
  return vec;
}

// src/math.ts
function matVecMul(mat, vec, rows, cols) {
  const out = new Float32Array(rows);
  for (let r = 0;r < rows; r++) {
    let sum = 0;
    const offset = r * cols;
    for (let c = 0;c < cols; c++) {
      sum += mat[offset + c] * vec[c];
    }
    out[r] = sum;
  }
  return out;
}
function vecAdd(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0;i < a.length; i++) {
    out[i] = a[i] + b[i];
  }
  return out;
}
function vecMul(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0;i < a.length; i++) {
    out[i] = a[i] * b[i];
  }
  return out;
}
function sigmoid(x) {
  const out = new Float32Array(x.length);
  for (let i = 0;i < x.length; i++) {
    out[i] = 1 / (1 + Math.exp(-x[i]));
  }
  return out;
}
function tanhVec(x) {
  const out = new Float32Array(x.length);
  for (let i = 0;i < x.length; i++) {
    out[i] = Math.tanh(x[i]);
  }
  return out;
}
function softplus(x) {
  const out = new Float32Array(x.length);
  for (let i = 0;i < x.length; i++) {
    const v = x[i];
    out[i] = v > 20 ? v : Math.log(1 + Math.exp(v));
  }
  return out;
}
function expVec(x) {
  const out = new Float32Array(x.length);
  for (let i = 0;i < x.length; i++) {
    out[i] = Math.exp(x[i]);
  }
  return out;
}
function softmax(x) {
  let max = -Infinity;
  for (let i = 0;i < x.length; i++) {
    if (x[i] > max)
      max = x[i];
  }
  const out = new Float32Array(x.length);
  let sum = 0;
  for (let i = 0;i < x.length; i++) {
    const v = Math.exp(x[i] - max);
    out[i] = v;
    sum += v;
  }
  for (let i = 0;i < x.length; i++) {
    out[i] = out[i] / sum;
  }
  return out;
}
function concat(...arrays) {
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
function vecAddScalar(x, s) {
  const out = new Float32Array(x.length);
  for (let i = 0;i < x.length; i++) {
    out[i] = x[i] + s;
  }
  return out;
}

// src/attention.ts
function attentionStep(prevWindow, x, h1, kappaPrev, charOneHots, charLen, alphabetSize, weights, bias, K) {
  const input = concat(prevWindow, x, h1);
  const inputSize = input.length;
  const outputSize = 3 * K;
  const raw = vecAdd(matVecMul(weights, input, outputSize, inputSize), bias);
  const params = softplus(raw);
  const alpha = params.subarray(0, K);
  const beta = params.subarray(K, 2 * K);
  const kappaDelta = params.subarray(2 * K, 3 * K);
  const kappa = new Float32Array(K);
  for (let k = 0;k < K; k++) {
    kappa[k] = kappaPrev[k] + kappaDelta[k] / 25;
  }
  for (let k = 0;k < K; k++) {
    if (beta[k] < 0.01)
      beta[k] = 0.01;
  }
  const window = new Float32Array(alphabetSize);
  for (let u = 0;u < charLen; u++) {
    let phi = 0;
    for (let k = 0;k < K; k++) {
      const diff = kappa[k] - u;
      phi += alpha[k] * Math.exp(-(diff * diff) / beta[k]);
    }
    const charOffset = u * alphabetSize;
    for (let j = 0;j < alphabetSize; j++) {
      window[j] = (window[j] ?? 0) + phi * charOneHots[charOffset + j];
    }
  }
  return [window, kappa];
}

// src/lstm.ts
function lstmStep(x, hPrev, cPrev, kernel, bias, numUnits) {
  const input = concat(x, hPrev);
  const inputSize = input.length;
  const gateSize = 4 * numUnits;
  const gates = vecAdd(matVecMul(kernel, input, gateSize, inputSize), bias);
  const iGate = sigmoid(gates.slice(0, numUnits));
  const jGate = tanhVec(gates.slice(numUnits, 2 * numUnits));
  const fGate = sigmoid(vecAddScalar(gates.slice(2 * numUnits, 3 * numUnits), 1));
  const oGate = sigmoid(gates.slice(3 * numUnits, 4 * numUnits));
  const cNew = vecAdd(vecMul(fGate, cPrev), vecMul(iGate, jGate));
  const hNew = vecMul(oGate, tanhVec(cNew));
  return [hNew, cNew];
}

// src/random.ts
var hasCached = false;
var cachedValue = 0;
function gaussRandom() {
  if (hasCached) {
    hasCached = false;
    return cachedValue;
  }
  let u;
  let v;
  let r;
  do {
    u = 2 * Math.random() - 1;
    v = 2 * Math.random() - 1;
    r = u * u + v * v;
  } while (r === 0 || r > 1);
  const c = Math.sqrt(-2 * Math.log(r) / r);
  cachedValue = v * c;
  hasCached = true;
  return u * c;
}
function sampleBivariateNormal(mu1, mu2, sigma1, sigma2, rho) {
  const z1 = gaussRandom();
  const z2 = gaussRandom();
  const x = Math.sqrt(1 - rho * rho) * sigma1 * z1 + rho * sigma1 * z2 + mu1;
  const y = sigma2 * z2 + mu2;
  return [x, y];
}
function sampleCategorical(probs) {
  const threshold = Math.random();
  let cumulative = 0;
  for (let i = 0;i < probs.length; i++) {
    cumulative += probs[i];
    if (cumulative >= threshold)
      return i;
  }
  return probs.length - 1;
}

// src/mdn.ts
function parseMdnOutput(raw, numComponents, bias) {
  const N = numComponents;
  const rawPis = raw.slice(0, N);
  const rawSigmas = raw.slice(N, 3 * N);
  const rawRhos = raw.slice(3 * N, 4 * N);
  const rawMus = raw.slice(4 * N, 6 * N);
  const rawEs = raw.slice(6 * N, 6 * N + 1);
  const biasedPis = new Float32Array(N);
  for (let i = 0;i < N; i++) {
    biasedPis[i] = rawPis[i] * (1 + bias);
  }
  const biasedSigmas = new Float32Array(2 * N);
  for (let i = 0;i < 2 * N; i++) {
    biasedSigmas[i] = rawSigmas[i] - bias;
  }
  const pis = softmax(biasedPis);
  const sigmas = expVec(biasedSigmas);
  const rhos = tanhVec(rawRhos);
  const esVal = 1 / (1 + Math.exp(-rawEs[0]));
  for (let i = 0;i < sigmas.length; i++) {
    if (sigmas[i] < 0.0001)
      sigmas[i] = 0.0001;
  }
  const eps = 0.00000001;
  for (let i = 0;i < rhos.length; i++) {
    const v = rhos[i];
    if (v < eps - 1)
      rhos[i] = eps - 1;
    else if (v > 1 - eps)
      rhos[i] = 1 - eps;
  }
  const eos = Math.max(eps, Math.min(1 - eps, esVal));
  const mu1 = rawMus.slice(0, N);
  const mu2 = rawMus.slice(N, 2 * N);
  const sigma1 = sigmas.slice(0, N);
  const sigma2 = sigmas.slice(N, 2 * N);
  for (let i = 0;i < N; i++) {
    if (pis[i] < 0.01)
      pis[i] = 0;
  }
  return { pis, mu1, mu2, sigma1, sigma2, rho: rhos, eos };
}
function sampleFromMixture(params) {
  const idx = sampleCategorical(params.pis);
  const [dx, dy] = sampleBivariateNormal(params.mu1[idx], params.mu2[idx], params.sigma1[idx], params.sigma2[idx], params.rho[idx]);
  const penUp = Math.random() < params.eos;
  return [dx, dy, penUp];
}

// src/model.ts
var NUM_UNITS = 400;
var NUM_ATTENTION_COMPONENTS = 10;
var NUM_MIXTURE_COMPONENTS = 20;
function getWeight(weights, name) {
  const w = weights[name];
  if (!w)
    throw new Error(`Missing weight tensor: ${name}`);
  return w;
}

class SynthesisModel {
  lstm1Kernel;
  lstm1Bias;
  lstm2Kernel;
  lstm2Bias;
  lstm3Kernel;
  lstm3Bias;
  attnWeights;
  attnBias;
  gmmWeights;
  gmmBias;
  constructor(weights) {
    this.lstm1Kernel = getWeight(weights, "lstm1_kernel").data;
    this.lstm1Bias = getWeight(weights, "lstm1_bias").data;
    this.lstm2Kernel = getWeight(weights, "lstm2_kernel").data;
    this.lstm2Bias = getWeight(weights, "lstm2_bias").data;
    this.lstm3Kernel = getWeight(weights, "lstm3_kernel").data;
    this.lstm3Bias = getWeight(weights, "lstm3_bias").data;
    this.attnWeights = getWeight(weights, "attention_weights").data;
    this.attnBias = getWeight(weights, "attention_bias").data;
    this.gmmWeights = getWeight(weights, "gmm_weights").data;
    this.gmmBias = getWeight(weights, "gmm_bias").data;
  }
  generate(text, options) {
    const bias = options?.bias ?? 0.5;
    const maxSteps = options?.maxSteps ?? 40 * text.length;
    const charIndices = encodeText(text);
    const charLen = charIndices.length;
    const charOneHots = new Float32Array(charLen * ALPHABET_SIZE);
    for (let i = 0;i < charLen; i++) {
      const oh = oneHot(charIndices[i], ALPHABET_SIZE);
      charOneHots.set(oh, i * ALPHABET_SIZE);
    }
    let h1 = new Float32Array(NUM_UNITS);
    let c1 = new Float32Array(NUM_UNITS);
    let h2 = new Float32Array(NUM_UNITS);
    let c2 = new Float32Array(NUM_UNITS);
    let h3 = new Float32Array(NUM_UNITS);
    let c3 = new Float32Array(NUM_UNITS);
    let kappa = new Float32Array(NUM_ATTENTION_COMPONENTS);
    let window = new Float32Array(ALPHABET_SIZE);
    let x = new Float32Array(3);
    x[2] = 1;
    const points = [];
    for (let step = 0;step < maxSteps; step++) {
      const s1In = concat(window, x);
      [h1, c1] = lstmStep(s1In, h1, c1, this.lstm1Kernel, this.lstm1Bias, NUM_UNITS);
      [window, kappa] = attentionStep(window, x, h1, kappa, charOneHots, charLen, ALPHABET_SIZE, this.attnWeights, this.attnBias, NUM_ATTENTION_COMPONENTS);
      const s2In = concat(x, h1, window);
      [h2, c2] = lstmStep(s2In, h2, c2, this.lstm2Kernel, this.lstm2Bias, NUM_UNITS);
      const s3In = concat(x, h2, window);
      [h3, c3] = lstmStep(s3In, h3, c3, this.lstm3Kernel, this.lstm3Bias, NUM_UNITS);
      const rawOutput = vecAdd(matVecMul(this.gmmWeights, h3, NUM_MIXTURE_COMPONENTS * 6 + 1, NUM_UNITS), this.gmmBias);
      const mdnParams = parseMdnOutput(rawOutput, NUM_MIXTURE_COMPONENTS, bias);
      const [dx, dy, penUp] = sampleFromMixture(mdnParams);
      points.push({ dx, dy, penUp });
      let maxPhiIdx = 0;
      let maxPhiVal = -Infinity;
      for (let u = 0;u < charLen; u++) {
        let phi = 0;
        for (let k = 0;k < NUM_ATTENTION_COMPONENTS; k++) {
          const diff = kappa[k] - u;
          phi += Math.exp(-diff * diff);
        }
        if (phi > maxPhiVal) {
          maxPhiVal = phi;
          maxPhiIdx = u;
        }
      }
      const pastFinalChar = maxPhiIdx >= charLen - 1;
      if (pastFinalChar && penUp)
        break;
      x = new Float32Array([dx, dy, penUp ? 1 : 0]);
    }
    return points;
  }
}

// src/svg.ts
function alignCoords(coords) {
  if (coords.length < 2)
    return coords;
  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumXY = 0;
  const n = coords.length;
  for (const c of coords) {
    sumX += c.x;
    sumY += c.y;
    sumXX += c.x * c.x;
    sumXY += c.x * c.y;
  }
  const denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 0.0000000001)
    return coords;
  const slope = (n * sumXY - sumX * sumY) / denom;
  const offset = (sumY - slope * sumX) / n;
  const theta = Math.atan(slope);
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  return coords.map((c) => ({
    x: cosT * c.x + sinT * c.y,
    y: -sinT * c.x + cosT * c.y - offset
  }));
}
function convertAndAlign(points) {
  const allCoords = [];
  let cx = 0;
  let cy = 0;
  for (const p of points) {
    cx += p.dx;
    cy += p.dy;
    allCoords.push({ x: cx, y: -cy, penUp: p.penUp });
  }
  const xyOnly = allCoords.map((c) => ({ x: c.x, y: c.y }));
  const aligned = alignCoords(xyOnly);
  for (let i = 0;i < allCoords.length; i++) {
    const a = aligned[i];
    const c = allCoords[i];
    c.x = a.x;
    c.y = a.y;
  }
  return allCoords;
}
function splitStrokes(allCoords) {
  const strokes = [];
  let current = [];
  for (const { x, y, penUp } of allCoords) {
    current.push({ x, y });
    if (penUp) {
      if (current.length > 0) {
        strokes.push(current);
        current = [];
      }
    }
  }
  if (current.length > 0) {
    strokes.push(current);
  }
  return strokes;
}
function smoothPath(points) {
  if (points.length === 0)
    return "";
  if (points.length === 1) {
    const p = points[0];
    return `M${p.x.toFixed(2)},${p.y.toFixed(2)}`;
  }
  if (points.length === 2) {
    const p0 = points[0];
    const p1 = points[1];
    return `M${p0.x.toFixed(2)},${p0.y.toFixed(2)} L${p1.x.toFixed(2)},${p1.y.toFixed(2)}`;
  }
  const first = points[0];
  let d = `M${first.x.toFixed(2)},${first.y.toFixed(2)}`;
  const p1 = points[1];
  const mx = (first.x + p1.x) / 2;
  const my = (first.y + p1.y) / 2;
  d += ` L${mx.toFixed(2)},${my.toFixed(2)}`;
  for (let i = 1;i < points.length - 1; i++) {
    const cp = points[i];
    const next = points[i + 1];
    const endX = (cp.x + next.x) / 2;
    const endY = (cp.y + next.y) / 2;
    d += ` Q${cp.x.toFixed(2)},${cp.y.toFixed(2)} ${endX.toFixed(2)},${endY.toFixed(2)}`;
  }
  const last = points[points.length - 1];
  d += ` L${last.x.toFixed(2)},${last.y.toFixed(2)}`;
  return d;
}
function strokesToSvgPaths(points) {
  if (points.length === 0)
    return [];
  const allCoords = convertAndAlign(points);
  const strokes = splitStrokes(allCoords);
  return strokes.map(smoothPath);
}
function strokesToCoords(points) {
  if (points.length === 0)
    return [];
  const allCoords = convertAndAlign(points);
  return splitStrokes(allCoords);
}
var DEFAULT_MIN_WIDTH = 0.1;
var DEFAULT_MAX_WIDTH = 1.2;
var DEFAULT_SPEED_CAP = 2;
var DEFAULT_SMOOTHING_WINDOW = 3;
function outlinePath(stroke, options) {
  if (stroke.length === 0)
    return "";
  if (stroke.length === 1) {
    const p = stroke[0];
    const r = (options?.maxWidth ?? DEFAULT_MAX_WIDTH) / 2 * 0.3;
    return `M${(p.x - r).toFixed(2)},${p.y.toFixed(2)} ` + `A${r.toFixed(2)},${r.toFixed(2)} 0 1,0 ${(p.x + r).toFixed(2)},${p.y.toFixed(2)} ` + `A${r.toFixed(2)},${r.toFixed(2)} 0 1,0 ${(p.x - r).toFixed(2)},${p.y.toFixed(2)}Z`;
  }
  const minW = options?.minWidth ?? DEFAULT_MIN_WIDTH;
  const maxW = options?.maxWidth ?? DEFAULT_MAX_WIDTH;
  const speedCap = options?.speedCap ?? DEFAULT_SPEED_CAP;
  const smoothWin = options?.smoothingWindow ?? DEFAULT_SMOOTHING_WINDOW;
  const n = stroke.length;
  const distances = [0];
  for (let i = 1;i < n; i++) {
    const prev = stroke[i - 1];
    const curr = stroke[i];
    const dx = curr.x - prev.x;
    const dy = curr.y - prev.y;
    distances.push(Math.sqrt(dx * dx + dy * dy));
  }
  const rawWidths = distances.map((d) => {
    const t = Math.min(d / speedCap, 1);
    return maxW + t * (minW - maxW);
  });
  const widths = [];
  const halfWin = Math.floor(smoothWin / 2);
  for (let i = 0;i < n; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - halfWin);j <= Math.min(n - 1, i + halfWin); j++) {
      sum += rawWidths[j];
      count++;
    }
    widths.push(sum / count);
  }
  const taperLen = Math.min(3, Math.floor(n / 2));
  for (let i = 0;i < taperLen; i++) {
    const t = (i + 1) / (taperLen + 1);
    widths[i] = widths[i] * t;
    widths[n - 1 - i] = widths[n - 1 - i] * t;
  }
  const normals = [];
  for (let i = 0;i < n; i++) {
    let tx;
    let ty;
    if (i === 0) {
      const next = stroke[1];
      const curr = stroke[0];
      tx = next.x - curr.x;
      ty = next.y - curr.y;
    } else if (i === n - 1) {
      const curr = stroke[n - 1];
      const prev = stroke[n - 2];
      tx = curr.x - prev.x;
      ty = curr.y - prev.y;
    } else {
      const next = stroke[i + 1];
      const prev = stroke[i - 1];
      tx = next.x - prev.x;
      ty = next.y - prev.y;
    }
    const len = Math.sqrt(tx * tx + ty * ty);
    if (len < 0.0000000001) {
      normals.push({ x: 0, y: 1 });
    } else {
      normals.push({ x: -ty / len, y: tx / len });
    }
  }
  const left = [];
  const right = [];
  for (let i = 0;i < n; i++) {
    const p = stroke[i];
    const nm = normals[i];
    const hw = widths[i] / 2;
    left.push({ x: p.x + nm.x * hw, y: p.y + nm.y * hw });
    right.push({ x: p.x - nm.x * hw, y: p.y - nm.y * hw });
  }
  let d = `M${left[0].x.toFixed(2)},${left[0].y.toFixed(2)}`;
  if (left.length === 2) {
    const p = left[1];
    d += ` L${p.x.toFixed(2)},${p.y.toFixed(2)}`;
  } else {
    for (let i = 1;i < left.length - 1; i++) {
      const cp = left[i];
      const next = left[i + 1];
      const endX = (cp.x + next.x) / 2;
      const endY = (cp.y + next.y) / 2;
      if (i === 1) {
        const prev = left[0];
        const fmx = (prev.x + cp.x) / 2;
        const fmy = (prev.y + cp.y) / 2;
        d += ` L${fmx.toFixed(2)},${fmy.toFixed(2)}`;
      }
      d += ` Q${cp.x.toFixed(2)},${cp.y.toFixed(2)} ${endX.toFixed(2)},${endY.toFixed(2)}`;
    }
    const lastL = left[left.length - 1];
    d += ` L${lastL.x.toFixed(2)},${lastL.y.toFixed(2)}`;
  }
  const lastR = right[right.length - 1];
  d += ` L${lastR.x.toFixed(2)},${lastR.y.toFixed(2)}`;
  if (right.length === 2) {
    const p = right[0];
    d += ` L${p.x.toFixed(2)},${p.y.toFixed(2)}`;
  } else {
    for (let i = right.length - 2;i > 0; i--) {
      const cp = right[i];
      const next = right[i - 1];
      const endX = (cp.x + next.x) / 2;
      const endY = (cp.y + next.y) / 2;
      if (i === right.length - 2) {
        const prev = right[right.length - 1];
        const fmx = (prev.x + cp.x) / 2;
        const fmy = (prev.y + cp.y) / 2;
        d += ` L${fmx.toFixed(2)},${fmy.toFixed(2)}`;
      }
      d += ` Q${cp.x.toFixed(2)},${cp.y.toFixed(2)} ${endX.toFixed(2)},${endY.toFixed(2)}`;
    }
    const firstR = right[0];
    d += ` L${firstR.x.toFixed(2)},${firstR.y.toFixed(2)}`;
  }
  d += "Z";
  return d;
}
function strokesToOutlinePaths(points, options) {
  const strokes = strokesToCoords(points);
  return strokes.map((s) => outlinePath(s, options));
}

// src/weights.ts
var MAGIC = 1347568467;
function loadWeights(buffer) {
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
  for (let t = 0;t < tensorCount; t++) {
    const nameLen = view.getUint16(offset, true);
    offset += 2;
    const nameBytes = new Uint8Array(buffer, offset, nameLen);
    const name = new TextDecoder().decode(nameBytes);
    offset += nameLen;
    const ndims = view.getUint8(offset);
    offset += 1;
    const shape = [];
    let totalElements = 1;
    for (let d = 0;d < ndims; d++) {
      const dim = view.getUint32(offset, true);
      offset += 4;
      shape.push(dim);
      totalElements *= dim;
    }
    const data = new Float32Array(totalElements);
    for (let i = 0;i < totalElements; i++) {
      data[i] = view.getFloat32(offset, true);
      offset += 4;
    }
    weights[name] = { data, shape };
  }
  return weights;
}
async function fetchWeights(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch weights from ${url}: ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  return loadWeights(buffer);
}

// src/index.ts
var cachedWeights = null;
var DEFAULT_WEIGHTS_URL = "synthesis.bin";
async function synthesize(text, options) {
  const validChars = new Set(ALPHABET);
  for (const ch of text) {
    if (!validChars.has(ch)) {
      throw new Error(`Invalid character '${ch}' (code ${ch.charCodeAt(0)}). ` + "Supported characters: letters, digits, and common punctuation.");
    }
  }
  if (!cachedWeights) {
    const url = options?.weightsUrl ?? DEFAULT_WEIGHTS_URL;
    cachedWeights = await fetchWeights(url);
  }
  const model = new SynthesisModel(cachedWeights);
  const generateOptions = {};
  if (options?.bias !== undefined)
    generateOptions.bias = options.bias;
  if (options?.maxSteps !== undefined)
    generateOptions.maxSteps = options.maxSteps;
  const points = model.generate(text, generateOptions);
  return strokesToSvgPaths(points);
}
export {
  ALPHABET,
  ALPHABET_SIZE,
  SynthesisModel,
  fetchWeights,
  loadWeights,
  outlinePath,
  strokesToCoords,
  strokesToOutlinePaths,
  strokesToSvgPaths,
  synthesize
};

//# debugId=FD0EFF3A4EB9583764756E2164756E21
//# sourceMappingURL=index.js.map
