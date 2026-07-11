// Pure helpers: decode the compact JSON encodings and do the small bits of
// math the renderer needs at scroll time. Everything heavy was precomputed
// offline (see generate/generate.py); here we only expand + interpolate.

// Expand the "ragged_q8" attention encoding into a dense [layer][head][q][k]
// float array. Causal rows store only keys 0..q; we pad the rest with 0.
export function decodeAttention(att, N) {
  if (att.encoding !== "ragged_q8") return att.weights; // assume dense floats
  const s = att.scale;
  return att.weights.map((heads) =>
    heads.map((rows) => {
      const dense = [];
      for (let q = 0; q < N; q++) {
        const row = new Array(N).fill(0);
        const r = rows[q] || [];
        for (let k = 0; k < r.length; k++) row[k] = r[k] * s;
        dense.push(row);
      }
      return dense;
    })
  );
}

// Softmax with temperature over a list of logits.
export function softmaxT(logits, T) {
  const t = Math.max(T, 1e-3);
  const m = Math.max(...logits);
  const ex = logits.map((l) => Math.exp((l - m) / t));
  const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map((e) => e / sum);
}

export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export const lerp = (a, b, t) => a + (b - a) * t;

export function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

// Display text for a token: whitespace-only tokens are stripped to "" by the
// generator — show a visible space mark instead of nothing.
export const displayToken = (t) => (t === "" ? "␣" : t);

// Map a 0..1 scroll progress to an integer index in [0, n-1].
export function progressToIndex(p, n) {
  return clamp(Math.round(p * (n - 1)), 0, n - 1);
}
