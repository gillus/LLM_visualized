// "From parrot to partner" — a scrollytelling tour of the three schools a
// language model goes through: pretraining (autocomplete), fine-tuning
// (assistant), reinforcement learning (more accurate, learns to think).
//
// All content is precomputed in web/training.json (real SmolLM2 base/instruct
// for stages 1–2, hand-authored for the RL stage). Nothing runs in the browser
// but a few class toggles and one tiny typewriter — scroll just chooses which
// sub-panel of each sticky card is showing.
import { initAaToggle, showLoadError } from "./aa.js";
import { mountGlossary } from "./glossary.js";

const RM = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};

let DATA = null;
const stages = {};

// ---------------------------------------------------------------------------
// shared bits
// ---------------------------------------------------------------------------
function chatQuestion() {
  return el("div", "chat-q", `🧒 ${DATA.prompt}`);
}
function verbatimTag(verbatim) {
  return verbatim
    ? `<span class="src-tag real aa-detail" title="output reale del modello">output reale del modello</span>`
    : `<span class="src-tag demo aa-detail" title="testo scritto a mano al posto del modello">illustrativo</span>`;
}

// a light typewriter for the parrot ramble; instant under reduced-motion
function typeInto(node, text) {
  node._timer && clearInterval(node._timer);
  if (RM) { node.textContent = text; return; }
  node.textContent = "";
  let i = 0;
  node._timer = setInterval(() => {
    node.textContent = text.slice(0, (i += 2));
    if (i >= text.length) { node.textContent = text; clearInterval(node._timer); }
  }, 18);
}

// only one named panel of a card visible at a time
function showPanel(root, name) {
  root.querySelectorAll("[data-panel]").forEach((p) =>
    p.classList.toggle("on", p.dataset.panel === name));
}

// ---------------------------------------------------------------------------
// ① PRETRAINING — the parrot
// ---------------------------------------------------------------------------
stages.parrot = {
  mount(root) {
    const card = el("div", "train-card");
    card.appendChild(chatQuestion());

    const intro = el("div", "panel-hint");
    intro.dataset.panel = "intro";
    intro.innerHTML = `Un modello appena nato, nutrito soltanto con testo
      grezzo. Vediamo cosa ha imparato a fare.`;

    const mix = el("div", "data-mix");
    mix.dataset.panel = "data";
    mix.appendChild(el("p", "mix-title", "La sua dieta di dati"));
    DATA.data_mix.forEach((m) => {
      const row = el("div", "mix-row");
      row.innerHTML = `<span class="mix-lbl">${m.label}</span>
        <span class="mix-track"><span class="mix-bar" style="width:${m.pct}%"></span></span>
        <span class="mix-pct">${m.pct}%</span>`;
      mix.appendChild(row);
    });
    mix.appendChild(el("p", "mix-note", "Nessuna etichetta, nessuna “risposta giusta” — solo testo."));

    const ask = el("div", "chat-a parrot");
    ask.dataset.panel = "ask";
    ask.innerHTML = `<span class="speaker">🦜 il modello grezzo continua a scrivere…</span>
      <span class="parrot-text"></span>
      ${verbatimTag(DATA.pretrain.verbatim)}
      <p class="card-caption">${DATA.pretrain.caption}</p>`;

    card.append(intro, mix, ask);
    root.appendChild(card);
    this.root = card;
    showPanel(card, "intro");
  },
  onEnter(step) {
    showPanel(this.root, step);
    if (step === "ask") typeInto($(".parrot-text", this.root), DATA.pretrain.answer);
  },
};

// ---------------------------------------------------------------------------
// ② FINE-TUNING — the helper
// ---------------------------------------------------------------------------
stages.helper = {
  mount(root) {
    const card = el("div", "train-card");

    const intro = el("div", "panel-hint");
    intro.dataset.panel = "intro";
    intro.innerHTML = `Stesso modello — adesso gli insegniamo la <em>forma</em>
      di una buona risposta, usando esempi scritti da persone.`;

    const cards = el("div", "sft-cards");
    cards.dataset.panel = "examples";
    cards.appendChild(el("p", "sft-title", "Lezioni di esempio (centinaia di migliaia)"));
    DATA.finetune.examples.forEach((ex) => {
      const c = el("div", "sft-card");
      c.innerHTML = `<div class="sft-q">🧑‍🏫 ${ex.instruction}</div>
        <div class="sft-r">✅ ${ex.response}</div>`;
      cards.appendChild(c);
    });

    const ans = el("div", "answer-panel");
    ans.dataset.panel = "answer";
    ans.appendChild(chatQuestion());
    const bubble = el("div", "chat-a helper");
    bubble.innerHTML = `<span class="speaker">🤖 l'assistente</span>
      <span class="answer-text">${DATA.finetune.answer}</span>
      ${verbatimTag(DATA.finetune.verbatim)}
      <p class="card-caption">${DATA.finetune.caption}</p>`;
    ans.appendChild(bubble);

    card.append(intro, cards, ans);
    root.appendChild(card);
    this.root = card;
    showPanel(card, "intro");
  },
  onEnter(step) { showPanel(this.root, step); },
};

// ---------------------------------------------------------------------------
// ③ REINFORCEMENT LEARNING — the expert
// ---------------------------------------------------------------------------
stages.expert = {
  mount(root) {
    const card = el("div", "train-card");

    const intro = el("div", "panel-hint");
    intro.dataset.panel = "intro";
    intro.innerHTML = `Basta copiare risposte. Ora il modello <em>ci
      prova</em>, e un voto gli dice se il tentativo è stato buono.`;

    // (a) you-be-the-reward-model
    const pref = el("div", "reward");
    pref.dataset.panel = "reward";
    const p = DATA.rl.preference;
    pref.appendChild(el("div", "reward-q", `🧒 ${p.question}`));
    pref.appendChild(el("p", "reward-ask", "Quale risposta premieresti?"));
    const opts = el("div", "reward-opts");
    p.candidates.forEach((cand, i) => {
      const b = el("button", "reward-opt");
      b.type = "button";
      b.dataset.i = i;
      b.innerHTML = `<span class="opt-text">${cand.text}</span>`;
      b.addEventListener("click", () => this.judge(i));
      opts.appendChild(b);
    });
    pref.appendChild(opts);
    pref.appendChild(el("div", "reward-verdict"));
    this.pref = pref;

    // (b) verifiable reward — the math thread
    const v = DATA.rl.verifiable;
    const math = el("div", "math");
    math.dataset.panel = "verifiable";
    math.appendChild(el("div", "math-problem", `🧮 ${v.problem}`));
    const naive = el("div", "math-naive");
    naive.innerHTML = `<span class="speaker">🦜 risposta veloce</span>
      <span class="naive-ans">${v.naive_answer}</span> <span class="grade bad">✗</span>
      <p class="card-caption">${v.naive_note}</p>`;
    math.appendChild(naive);
    const btn = el("button", "think-btn", "Lascialo ragionare →");
    btn.type = "button";
    btn.addEventListener("click", () => this.think());
    math.appendChild(btn);
    const steps = el("ol", "math-steps");
    v.thinking.forEach((s) => steps.appendChild(el("li", null, s)));
    math.appendChild(steps);
    const got = el("div", "math-answer");
    got.innerHTML = `<span class="grade good">✓</span> <strong>${v.answer}</strong>`;
    math.appendChild(got);
    math.appendChild(el("p", "card-caption math-reward", v.reward_note));

    card.append(intro, pref, math);
    root.appendChild(card);
    this.root = card;
    showPanel(card, "intro");
  },

  judge(i) {
    const p = DATA.rl.preference;
    if (this.pref.classList.contains("decided")) return;
    this.pref.classList.add("decided");
    this.pref.querySelectorAll(".reward-opt").forEach((b) => {
      const idx = +b.dataset.i;
      const cand = p.candidates[idx];
      b.classList.add(cand.better ? "correct" : "wrong");
      if (idx === i) b.classList.add("chosen");
      b.insertAdjacentHTML("beforeend",
        `<span class="opt-tag">${cand.tag}</span>
         <span class="opt-reward">${cand.better ? "▲ " + p.reward_up : "▼ " + p.reward_down}</span>`);
    });
    const right = p.candidates[i].better;
    $(".reward-verdict", this.pref).innerHTML =
      `<p class="verdict-head ${right ? "good" : "miss"}">${
        right ? "Questa è quella che gli addestratori premiano. 🎯" : "Allettante — ma guarda meglio."}</p>
       <p>${p.reveal}</p>`;
    $(".reward-verdict", this.pref).classList.add("on");
  },

  think() {
    const math = this.root.querySelector(".math");
    if (math.classList.contains("thought")) return;
    math.classList.add("thought");
    const steps = math.querySelectorAll(".math-steps li");
    const finish = () => {
      math.querySelector(".math-answer").classList.add("on");
      math.querySelector(".math-reward").classList.add("on");
    };
    if (RM) { steps.forEach((s) => s.classList.add("on")); finish(); return; }
    steps.forEach((s, k) => setTimeout(() => s.classList.add("on"), 260 * (k + 1)));
    setTimeout(finish, 260 * (steps.length + 1));
  },

  onEnter(step) { showPanel(this.root, step); },
};

// ---------------------------------------------------------------------------
// outro: the three report cards (same question, grown up)
// ---------------------------------------------------------------------------
function buildReportCards() {
  const wrap = $("#report-cards");
  if (!wrap) return;
  const better = DATA.rl.preference.candidates.find((c) => c.better);
  const answers = {
    parrot: DATA.pretrain.answer,
    helper: DATA.finetune.answer,
    expert: better ? better.text : DATA.rl.verifiable.answer,
  };
  DATA.report_cards.forEach((rc) => {
    const c = el("div", `report-card ${rc.kind}`);
    c.innerHTML = `<div class="rc-stage">${rc.stage}</div>
      <div class="rc-answer">${answers[rc.kind] || ""}</div>
      <div class="rc-verdict">${rc.verdict}</div>`;
    wrap.appendChild(c);
  });
}

// ---------------------------------------------------------------------------
// scrollama wiring (mirrors js/main.js)
// ---------------------------------------------------------------------------
function initScroller() {
  const scroller = scrollama();
  window.__scroller = scroller;
  scroller
    .setup({ step: ".step", offset: 0.62 })
    .onStepEnter((r) => {
      r.element.classList.add("is-active");
      const { stage, step } = r.element.dataset;
      stages[stage]?.onEnter?.(step);
      document.querySelectorAll(".rail-link").forEach((a) =>
        a.classList.toggle("active", a.dataset.stage === stage));
    })
    .onStepExit((r) => r.element.classList.remove("is-active"));
  window.addEventListener("resize", () => scroller.resize());
  document.fonts?.ready.then(() => scroller.resize());
}

async function main() {
  initAaToggle();
  mountGlossary();
  DATA = await (await fetch("training.json")).json();
  // gates the .vb-real / .vb-demo copy: only claim "real model output" when
  // the data actually is (see generate/training.py --fixture)
  document.body.classList.toggle("data-verbatim",
    !!(DATA.pretrain.verbatim && DATA.finetune.verbatim));
  for (const [id, stage] of Object.entries(stages)) {
    const root = document.querySelector(`[data-viz="${id}"]`);
    if (root) stage.mount(root);
  }
  buildReportCards();
  initScroller();
}

main().catch(showLoadError);
