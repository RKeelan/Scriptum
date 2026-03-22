// Box-Muller transform with caching (adapted from Distill/Karpathy)
let hasCached = false;
let cachedValue = 0;

/** Sample from standard normal distribution N(0,1). */
export function gaussRandom(): number {
  if (hasCached) {
    hasCached = false;
    return cachedValue;
  }
  let u: number;
  let v: number;
  let r: number;
  do {
    u = 2 * Math.random() - 1;
    v = 2 * Math.random() - 1;
    r = u * u + v * v;
  } while (r === 0 || r > 1);
  const c = Math.sqrt((-2 * Math.log(r)) / r);
  cachedValue = v * c;
  hasCached = true;
  return u * c;
}

/** Sample from bivariate normal distribution. Returns [x, y]. */
export function sampleBivariateNormal(
  mu1: number,
  mu2: number,
  sigma1: number,
  sigma2: number,
  rho: number,
): [number, number] {
  const z1 = gaussRandom();
  const z2 = gaussRandom();
  const x = Math.sqrt(1 - rho * rho) * sigma1 * z1 + rho * sigma1 * z2 + mu1;
  const y = sigma2 * z2 + mu2;
  return [x, y];
}

/** Sample an index from a categorical distribution (array of probabilities summing to 1). */
export function sampleCategorical(probs: Float32Array): number {
  const threshold = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i] as number;
    if (cumulative >= threshold) return i;
  }
  return probs.length - 1;
}
