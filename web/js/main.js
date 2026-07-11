// Orchestrator: loads a precomputed example, mounts the five stage renderers,
// wires Scrollama so scroll position drives state, and keeps the sticky token
// strip in sync. All stages read from one shared `app` object.
import { decodeAttention } from "./decode.js";
import { initAaToggle, showLoadError } from "./aa.js";
import { mountGlossary } from "./glossary.js";
import { createTokensStage } from "./stages/tokens.js";
import { createEmbeddingStage } from "./stages/embedding.js";
import { createLayersStage } from "./stages/layers.js";
import { createAttentionStage } from "./stages/attention.js";
import { createPredictionStage } from "./stages/prediction.js";

const PALETTE = ["#2563eb", "#db2777", "#16a34a", "#d97706", "#7c3aed",
  "#0891b2", "#dc2626", "#4f46e5", "#ca8a04", "#0d9488"];

const app = {
  data: null,
  attn: null,
  state: {},
  stages: {},
  colorForPos(pos) { return PALETTE[pos % PALETTE.length]; },
  set(partial) {
    Object.assign(this.state, partial);
    this.render();
  },
  render() {
    for (const s of Object.values(this.stages)) s.update?.(app);
    renderTokenStrip(app);
  },
};

async function loadExample(id) {
  const res = await fetch(`examples/${id}.json`);
  app.data = await res.json();
  app.attn = decodeAttention(app.data.attention, app.data.tokens.length);
  // default state: inspect the last token (where the real prediction happens)
  const lastPos = app.data.tokens.length - 1;
  app.state = {
    pos: lastPos, layer: 0, attnLayer: 0, head: 0, temp: 1.0,
    tokenReveal: 0, embStep: 0, showGrid: false, revealed: null,
    chain: 0,        // continuation words added via "add & predict again"
    posPinned: false, // true once the learner picks a token — story stops auto-walking it
    // tuned reads out far more cleanly mid-stack — default to it when present
    lensMode: app.data.logit_lens_tuned ? "tuned" : "plain",
  };
  for (const s of Object.values(app.stages)) {
    const root = document.querySelector(`[data-viz="${s.id}"]`);
    s.mount(root, app);
  }
  renderTokenStrip(app);
}

// ---- sticky token strip (context anchor + token selector) ----
function renderTokenStrip(app) {
  const strip = document.getElementById("token-strip");
  strip.innerHTML = "";
  app.data.tokens.forEach((t) => {
    const chip = document.createElement("button");
    chip.className = "strip-chip" + (t.pos === app.state.pos ? " selected" : "");
    chip.style.setProperty("--c", app.colorForPos(t.pos));
    chip.textContent = t.text;
    chip.title = `token #${t.id} (posizione ${t.pos}) — clicca per ispezionare`;
    chip.onclick = () => app.set({ pos: t.pos, posPinned: true });
    strip.appendChild(chip);
  });
  // words the model has added via "add & predict again" — display-only
  // (the other stages only have data for the original prompt tokens)
  (app.data.continuation || []).slice(0, app.state.chain ?? 0).forEach((c) => {
    const chip = document.createElement("span");
    chip.className = "strip-chip grown";
    chip.textContent = c.token.text;
    chip.title = "aggiunta dal modello (la sua parola più sicura)";
    strip.appendChild(chip);
  });
  const arrow = document.createElement("span");
  arrow.className = "strip-arrow";
  arrow.textContent = "→";
  strip.appendChild(arrow);
  const q = document.createElement("div");
  q.className = "strip-q" + (app.state.revealed ? " revealed" : "");
  q.textContent = app.state.revealed ?? "?";
  strip.appendChild(q);
}

// ---- header dropdown ----
async function initHeader() {
  const manifest = await (await fetch("examples/manifest.json")).json();
  const sel = document.getElementById("example-select");
  manifest.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.title;
    sel.appendChild(opt);
  });
  sel.onchange = async () => {
    await loadExample(sel.value);
    if (window.__scroller) window.__scroller.resize();
  };
  return manifest[0]?.id;
}

// ---- scrollama wiring ----
function initScroller() {
  const scroller = scrollama();
  window.__scroller = scroller;
  scroller
    .setup({ step: ".step", offset: 0.6, progress: true })
    .onStepEnter((r) => {
      r.element.classList.add("is-active");
      const { stage, step } = r.element.dataset;
      app.stages[stage]?.onEnter?.(step, app);
      setActiveRail(stage);
    })
    .onStepExit((r) => r.element.classList.remove("is-active"))
    .onStepProgress((r) => {
      const { stage, step } = r.element.dataset;
      app.stages[stage]?.onProgress?.(step, r.progress, app);
    });
  window.addEventListener("resize", () => scroller.resize());
  // webfonts (Fraunces etc.) swap in after first layout — re-measure the steps
  document.fonts?.ready.then(() => scroller.resize());
}

function setActiveRail(stage) {
  document.querySelectorAll(".rail-link").forEach((a) =>
    a.classList.toggle("active", a.dataset.stage === stage));
}

async function main() {
  app.stages = {
    tokens: createTokensStage(),
    embedding: createEmbeddingStage(),
    layers: createLayersStage(),
    attention: createAttentionStage(),
    prediction: createPredictionStage(),
  };
  initAaToggle();
  mountGlossary();   // before any fetch — the word list works even offline
  const firstId = await initHeader();
  await loadExample(firstId);
  initScroller();
}

main().catch(showLoadError);
