// Box-Muller transform with caching (adapted from Distill/Karpathy)
let hasCached = false;
let cachedValue = 0;
/** Sample from standard normal distribution N(0,1). */
export function gaussRandom() {
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
    const c = Math.sqrt((-2 * Math.log(r)) / r);
    cachedValue = v * c;
    hasCached = true;
    return u * c;
}
/** Sample from bivariate normal distribution. Returns [x, y]. */
export function sampleBivariateNormal(mu1, mu2, sigma1, sigma2, rho) {
    const z1 = gaussRandom();
    const z2 = gaussRandom();
    const x = Math.sqrt(1 - rho * rho) * sigma1 * z1 + rho * sigma1 * z2 + mu1;
    const y = sigma2 * z2 + mu2;
    return [x, y];
}
/** Sample an index from a categorical distribution (array of probabilities summing to 1). */
export function sampleCategorical(probs) {
    const threshold = Math.random();
    let cumulative = 0;
    for (let i = 0; i < probs.length; i++) {
        cumulative += probs[i];
        if (cumulative >= threshold)
            return i;
    }
    return probs.length - 1;
}
//# sourceMappingURL=random.js.map