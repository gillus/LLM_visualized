// Architecture page — two views over the SAME precomputed example JSON:
//   A. anatomy block diagram (static + expand one layer)
//   B. contamination lanes (token "ink" mixing across layers, scrubbed step by step)
// No new data: reuses logit_lens (per-layer guesses) and attention (mixing).
import { decodeAttention, displayToken } from "./decode.js";
import { showLoadError } from "./aa.js";
import { mountGlossary } from "./glossary.js";

// same token palette as the story page (main.js) — lanes must match the strip
const PALETTE = ["#2563eb", "#db2777", "#16a34a", "#d97706", "#7c3aed",
  "#0891b2", "#dc2626", "#4f46e5", "#ca8a04", "#0d9488"];
const colorFor = (pos) => PALETTE[pos % PALETTE.length];

const app = { data: null, attn: null };

// ---------- load ----------
async function loadExample(id) {
  app.data = await (await fetch(`examples/${id}.json`)).json();
  const N = app.data.tokens.length;
  app.attn = decodeAttention(app.data.attention, N);
  renderBlocks();
  renderFlow();
}

// ---------- A. block diagram ----------
function block(label, sub, cls = "") {
  return `<div class="ablock ${cls}"><span class="ablabel">${label}</span>` +
    (sub ? `<span class="absub">${sub}</span>` : "") + "</div>";
}
function arrow() { return `<div class="aarrow">▼</div>`; }

function renderBlocks() {
  const m = app.data.model;
  const el = document.getElementById("blocks");
  el.innerHTML =
    block(`Testo in ingresso`, `“${app.data.prompt}”`) + arrow() +
    block(`Tokenizzatore`, `testo → ${app.data.tokens.length} id di token`) + arrow() +
    block(`Embedding`, `ogni id → un vettore di ${m.d_model} dimensioni` +
      (m.tied_embeddings ? " (condiviso con l'output)" : "")) + arrow() +
    `<div class="ablock layerblock" id="layerblock">
       <div class="layerblock-head">
         <span class="ablabel">Strato Transformer <span class="xcount">× ${m.n_layers}</span></span>
         <span class="absub">clicca per espandere ▾</span>
       </div>
       <div class="layer-inner">
         ${block("RMSNorm", "")}${arrow()}
         ${block("Attenzione multi-testa",
            `${m.n_heads} teste di query · ${m.n_kv_heads} teste key/value (GQA) · posizioni RoPE · causale`, "sub")}
         ${block("+ residuo", "riaggiunge l'input", "thin")}${arrow()}
         ${block("RMSNorm", "")}${arrow()}
         ${block("Feed-forward (MLP)", `SwiGLU · ${m.d_model} → più grande → ${m.d_model}`, "sub")}
         ${block("+ residuo", "riaggiunge l'input", "thin")}
       </div>
     </div>` + arrow() +
    block(`RMSNorm finale`, ``) + arrow() +
    block(`Unembedding`, `vettore → ${m.vocab_size.toLocaleString()} punteggi, uno per parola`) + arrow() +
    block(`Softmax → Probabilità`, `punteggi → probabilità che sommano al 100%`, "out");

  document.getElementById("layerblock").querySelector(".layerblock-head")
    .onclick = () => document.getElementById("layerblock").classList.toggle("expanded");
}

// ---------- B. contamination lanes ----------
// Every token starts as pure "ink" of its own colour. At each layer, attention
// lets it absorb a little of the tokens before it. We measure the mix with
// attention rollout (Abnar & Zuidema 2020): compose (A + I)/2 layer by layer,
// where A is the head-mean attention. One correction: attention parked on the
// position-0 "attention sink" counts as SELF, not as flow from the first word.
// Idle heads park leftover attention there and its value contribution is
// ~nothing, so that share of the update is a no-op — the residual stream
// passes the token through unchanged. (Renormalizing the row instead would
// amplify tiny lookbacks and collapse every lane into the earliest token.)
function renderFlow() {
  const N = app.data.tokens.length;
  const tokens = app.data.tokens;
  // per-layer guesses for labels/hover: prefer the tuned lens (same default as
  // the story page) — the plain lens is noisy mid-stack
  const lens = app.data.logit_lens_tuned || app.data.logit_lens;
  const nAttn = app.attn.length;     // attention layers (30 real / 12 fixture)
  const nRows = nAttn + 1;           // + embedding row at the bottom

  // --- rollout: mix[r][q][k] = share of token q's ink from token k at row r
  const ident = () => tokens.map((_, q) =>
    tokens.map((_, k) => (k === q ? 1 : 0)));
  const mix = [ident()];
  for (let L = 0; L < nAttn; L++) {
    const heads = app.attn[L], H = heads.length;
    const prev = mix[L], next = [];
    for (let q = 0; q < N; q++) {
      const row = new Array(N).fill(0);
      if (q === 0) { row[0] = 1; next.push(row); continue; }
      // head-mean attention row (renormalized for int8 quantization drift)
      const m = new Array(N).fill(0);
      let tot = 0;
      for (let k = 0; k <= q; k++) {
        let v = 0;
        for (let h = 0; h < H; h++) v += heads[h][q][k];
        m[k] = v / H;
        tot += m[k];
      }
      for (let k = 0; k <= q; k++) m[k] /= tot || 1;
      // (A + I)/2, with the sink share m[0] folded into self
      for (let k = 1; k <= q; k++) {
        const w = 0.5 * m[k] + (k === q ? 0.5 + 0.5 * m[0] : 0);
        if (!w) continue;
        for (let s = 0; s <= k; s++) row[s] += w * prev[k][s];
      }
      next.push(row);
    }
    mix.push(next);
  }

  // --- sharpest cross-token lookback per (layer, query): strongest single
  // head reading a non-trivial target (not the sink, not the token itself)
  const edges = [];   // edges[L] = [{q, k, h, w}]
  for (let L = 0; L < nAttn; L++) {
    const heads = app.attn[L], list = [];
    for (let q = 2; q < N; q++) {
      let bw = 0, bk = -1, bh = -1;
      heads.forEach((m, h) => {
        for (let k = 1; k < q; k++)
          if (m[q][k] > bw) { bw = m[q][k]; bk = k; bh = h; }
      });
      if (bk >= 0 && bw >= 0.15) list.push({ q, k: bk, h: bh, w: bw });
    }
    edges.push(list);
  }

  // --- layout
  const cw = Math.max(40, Math.min(80, 560 / N));
  const rowH = Math.max(10, Math.min(16, 470 / nRows));
  const mL = 52, mR = 12, mT = 38, mB = 52;
  const W = mL + N * cw + mR, H = mT + nRows * rowH + mB;
  const colX = (pos) => mL + pos * cw + cw / 2;
  const rowY = (r) => mT + (nRows - 1 - r) * rowH + rowH / 2;  // r=0 bottom
  const laneW = Math.min(cw - 12, 46);

  const root = d3.select("#flow").html("");
  const svg = root.append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("class", "flow-svg");
  const info = document.getElementById("flow-info");

  // predicted next word per column (payoff at the top) + actual tokens below
  const gPred = svg.append("g");
  tokens.forEach((t, pos) => {
    const pred = lens[pos][nRows - 1][0].text;
    gPred.append("text").attr("x", colX(pos)).attr("y", mT - 16)
      .attr("class", "flow-pred").attr("text-anchor", "middle")
      .text("→ " + displayToken(pred || "·"));
    svg.append("text").attr("x", colX(pos)).attr("y", H - mB + 20)
      .attr("class", "flow-tok").attr("text-anchor", "middle")
      .attr("fill", colorFor(pos)).text(t.text);
  });
  svg.append("text").attr("x", mL).attr("y", H - mB + 40)
    .attr("class", "flow-axis").text("ogni corsia rappresenta un token — i suoi colori mostrano di chi è l'inchiostro contenuto");

  // layer labels (every 5)
  for (let r = 0; r < nRows; r++) {
    if (r % 5 === 0 || r === nRows - 1) {
      svg.append("text").attr("x", mL - 8).attr("y", rowY(r) + 3)
        .attr("class", "flow-rowlbl").attr("text-anchor", "end")
        .text(r === 0 ? "emb" : "S" + r);
    }
  }

  // ribbons (one group per layer, revealed by the scrubber)
  const gRibs = [];
  for (let L = 0; L < nAttn; L++) {
    const g = svg.append("g");
    for (const e of edges[L]) {
      g.append("path")
        .attr("d", `M ${colX(e.k)} ${rowY(L)} C ${colX(e.k)} ${rowY(L) - rowH * 1.6}, ` +
                   `${colX(e.q)} ${rowY(L + 1) + rowH * 1.6}, ${colX(e.q)} ${rowY(L + 1)}`)
        .attr("class", "lane-rib")
        .attr("stroke", colorFor(e.k))
        .attr("stroke-width", 1 + e.w * 6);
    }
    gRibs.push(g);
  }

  // lane cells: one stacked colour bar per (row, token)
  const gRows = [];
  for (let r = 0; r < nRows; r++) {
    const g = svg.append("g").attr("class", "lane-row");
    for (let q = 0; q < N; q++) {
      const x0 = colX(q) - laneW / 2, y0 = rowY(r) - (rowH - 3) / 2;
      let acc = 0;
      for (let k = 0; k <= q; k++) {
        const f = mix[r][q][k];
        if (f < 0.005) continue;
        g.append("rect")
          .attr("x", x0 + acc * laneW).attr("y", y0)
          .attr("width", Math.max(f * laneW, 0.5)).attr("height", rowH - 3)
          .attr("fill", colorFor(k));
        acc += f;
      }
      // hover catcher on top of the stack
      g.append("rect")
        .attr("x", x0).attr("y", y0).attr("width", laneW).attr("height", rowH - 3)
        .attr("class", "lane-hover")
        .on("mouseenter", () => showInfo(q, r))
        .on("mouseleave", resetInfo);
    }
    gRows.push(g);
  }

  function showInfo(q, r) {
    const parts = mix[r][q]
      .map((f, k) => ({ f, k }))
      .filter((d) => d.f >= 0.01)
      .sort((a, b) => b.f - a.f)
      .slice(0, 5)
      .map((d) => `<b style="color:${colorFor(d.k)}">${Math.round(d.f * 100)}% ${
        displayToken(tokens[d.k].text)}</b>`)
      .join(" · ");
    const g = lens[q][r][0];
    const where = r === 0 ? "all'inizio" : `dopo lo strato ${r}`;
    info.innerHTML = `<b>${tokens[q].text}</b> ${where} — composto da: ${parts}. ` +
      `La sua previsione per la parola successiva qui: “${displayToken(g.text || "·")}” (${(g.prob * 100).toFixed(1)}%).`;
  }
  function resetInfo() {
    info.textContent = "Passa il mouse su una barra per vedere di chi è l'inchiostro al suo interno…";
  }

  // --- scrubber
  const ctrl = d3.select("#flow-controls").html("");
  const back = ctrl.append("button").attr("type", "button").text("◀");
  const slider = ctrl.append("input")
    .attr("type", "range").attr("min", 0).attr("max", nAttn)
    .attr("value", 0).attr("aria-label", "strato");
  const fwd = ctrl.append("button").attr("type", "button").text("▶");
  const label = ctrl.append("span").attr("class", "lane-step-label");

  let step = 0;
  function draw(s) {
    step = Math.max(0, Math.min(s, nAttn));
    slider.property("value", step);
    gRows.forEach((g, r) => g.classed("future", r > step));
    gRibs.forEach((g, L) =>
      g.attr("display", L + 1 > step ? "none" : null)
       .attr("opacity", L + 1 === step ? 0.75 : 0.1));
    gPred.attr("opacity", step === nAttn ? 1 : 0.2);
    label.text(step === 0 ? "inizio — ogni token è puro"
      : step === nAttn ? `strato ${step} / ${nAttn} — completamente mescolato`
      : `dopo lo strato ${step} / ${nAttn}`);
  }
  back.on("click", () => draw(step - 1));
  fwd.on("click", () => draw(step + 1));
  slider.on("input", function () { draw(+this.value); });

  draw(0);
  resetInfo();

  // legend: lane colours = the tokens
  const leg = d3.select("#flow-legend").html("");
  tokens.forEach((t, pos) => {
    leg.append("span").attr("class", "leg-chip")
      .style("background", colorFor(pos)).text(displayToken(t.text));
  });
}

// ---------- header ----------
async function initHeader() {
  const manifest = await (await fetch("examples/manifest.json")).json();
  const sel = document.getElementById("example-select");
  manifest.forEach((m) => {
    const o = document.createElement("option");
    o.value = m.id; o.textContent = m.title; sel.appendChild(o);
  });
  sel.onchange = () => loadExample(sel.value);
  return manifest[0]?.id;
}

(async function () {
  mountGlossary();
  const first = await initHeader();
  await loadExample(first);
})().catch(showLoadError);
