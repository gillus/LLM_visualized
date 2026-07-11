// Stage 5 — Prediction. The final next-token distribution as a bar chart, with
// a temperature control (scroll-driven): cold = always the safest word, hot =
// more surprising. A dice button actually SAMPLES from the current distribution
// (over the full stored top list, not just the bars shown), so temperature is
// demonstrated, not just claimed. The "add & predict again" button walks the
// precomputed greedy continuation chain (data.continuation, see SCHEMA.md) —
// the autoregressive loop, shown instead of told. The final scroll step
// reveals the winner in the sticky token strip.
import { escapeHtml, displayToken } from "../decode.js";

const TOP_SHOWN = 8;
const ROLLS_KEPT = 10;
const T_MIN = 0.3, T_MAX = 1.8;

// the distribution currently on the table: the base prediction, or — after
// growing the sentence — the latest continuation step's
function currentTop(app) {
  const c = app.state.chain ?? 0;
  return c === 0 ? app.data.prediction.top : app.data.continuation[c - 1].top;
}

// Softmax(logit ÷ T) over the FULL vocabulary, not a renormalized top-50
// (which would inflate every percentage — "bed" would show 12.8% instead of
// its true 7.2%). We only store the top 50, so the tail — every other word —
// is approximated as (vocab − 50) equal slivers of the true leftover mass.
// Exact at T = 1, and right in both limits: cold → the winner takes all,
// hot → the tail swallows the distribution (why hot sampling goes strange).
function distWithTail(top, T, vocabSize) {
  const t = Math.max(T, 1e-3);
  const logits = top.map((d) => d.logit);
  const m = Math.max(...logits);
  const w = logits.map((l) => Math.exp((l - m) / t));
  const mass = top.reduce((a, d) => a + (d.prob ?? 0), 0);
  const nOther = Math.max((vocabSize ?? 0) - top.length, 0);
  let wTail = 0;
  if (nOther > 0 && top[0].prob > 0 && mass < 0.999) {
    // per-word tail logit, recovered from softmax(l₀) = p₀ at T = 1
    const tailLogit = top[0].logit - Math.log(top[0].prob) +
      Math.log((1 - mass) / nOther);
    wTail = nOther * Math.exp((tailLogit - m) / t);
  }
  const sum = w.reduce((a, b) => a + b, 0) + wTail;
  return { probs: w.map((v) => v / sum), tail: wTail > 0 ? wTail / sum : null };
}

export function createPredictionStage() {
  let elTempLabel, elTempFlavor, tempSlider, elBars, elRolls, elNote, rolls, lastRoll;
  let growBtn, resetBtn, elChain;

  function doRoll(app) {
    const top = currentTop(app);
    const T = app.state.temp ?? 1.0;
    const { probs, tail } = distWithTail(top, T, app.data.model?.vocab_size);
    // sample the same distribution the bars show — including the lumped tail,
    // so hot rolls really do wander off the chart
    const all = tail == null ? probs : [...probs, tail];
    let r = Math.random(), i = 0;
    while (i < all.length - 1 && (r -= all[i]) > 0) i++;
    if (tail != null && i === all.length - 1) {
      lastRoll = { i: -1, text: "…una parola più rara", prob: tail, tail: true };
      rolls.unshift(lastRoll);
      if (rolls.length > ROLLS_KEPT) rolls.pop();
      app.set({ revealed: null });   // we don't store which word it was
      return;
    }
    lastRoll = { i, text: top[i].text, prob: probs[i] };
    rolls.unshift(lastRoll);
    if (rolls.length > ROLLS_KEPT) rolls.pop();
    // the rolled word fills the "?" slot in the sticky strip
    app.set({ revealed: top[i].text });
  }

  function grow(app) {
    const c = app.state.chain ?? 0;
    if (c >= (app.data.continuation?.length ?? 0)) return;
    rolls = [];
    lastRoll = null;
    app.set({ chain: c + 1, revealed: null });
  }

  function resetChain(app) {
    rolls = [];
    lastRoll = null;
    app.set({ chain: 0, revealed: null });
  }

  return {
    id: "prediction",
    mount(root, app) {
      root.innerHTML = "";
      rolls = [];
      lastRoll = null;
      growBtn = resetBtn = elChain = null;
      const wrap = document.createElement("div");
      wrap.className = "pred-stage";
      // temperature header with a manual slider (built once — rebuilding it
      // every update would break an in-progress drag)
      const elTemp = document.createElement("div");
      elTemp.className = "pred-temp";
      elTempLabel = document.createElement("span");
      elTempLabel.className = "temp-label";
      tempSlider = document.createElement("input");
      tempSlider.type = "range";
      tempSlider.min = T_MIN;
      tempSlider.max = T_MAX;
      tempSlider.step = 0.01;
      tempSlider.setAttribute("aria-label", "temperatura");
      tempSlider.oninput = () => app.set({ temp: +tempSlider.value });
      elTempFlavor = document.createElement("span");
      elTempFlavor.className = "temp-flavor";
      elTemp.appendChild(elTempLabel);
      elTemp.appendChild(tempSlider);
      elTemp.appendChild(elTempFlavor);
      elBars = document.createElement("div");
      elBars.className = "pred-bars";
      wrap.appendChild(elTemp);
      wrap.appendChild(elBars);

      // the loop button — only when this example carries a continuation chain
      if (app.data.continuation?.length) {
        const growRow = document.createElement("div");
        growRow.className = "pred-grow";
        growBtn = document.createElement("button");
        growBtn.type = "button";
        growBtn.className = "grow-btn";
        growBtn.onclick = () => grow(app);
        elChain = document.createElement("span");
        elChain.className = "chain-status";
        resetBtn = document.createElement("button");
        resetBtn.type = "button";
        resetBtn.className = "reset-btn";
        resetBtn.textContent = "ricomincia da capo";
        resetBtn.onclick = () => resetChain(app);
        growRow.appendChild(growBtn);
        growRow.appendChild(elChain);
        growRow.appendChild(resetBtn);
        wrap.appendChild(growRow);
      }

      const rollBar = document.createElement("div");
      rollBar.className = "pred-roll";
      const dice = document.createElement("button");
      dice.type = "button";
      dice.className = "dice-btn";
      // inline die icon — emoji fonts are missing on some classroom machines
      dice.innerHTML =
        `<svg class="die" viewBox="0 0 24 24" aria-hidden="true">` +
        `<rect x="2" y="2" width="20" height="20" rx="5" fill="none" stroke="currentColor" stroke-width="2.2"/>` +
        `<circle cx="8" cy="8" r="1.9" fill="currentColor"/><circle cx="16" cy="8" r="1.9" fill="currentColor"/>` +
        `<circle cx="12" cy="12" r="1.9" fill="currentColor"/>` +
        `<circle cx="8" cy="16" r="1.9" fill="currentColor"/><circle cx="16" cy="16" r="1.9" fill="currentColor"/>` +
        `</svg>Lancia il dado per una parola`;
      dice.title = "Campiona una parola dalle probabilità attuali";
      dice.onclick = () => doRoll(app);
      elRolls = document.createElement("div");
      elRolls.className = "roll-history";
      rollBar.appendChild(dice);
      rollBar.appendChild(elRolls);

      elNote = document.createElement("div");
      elNote.className = "pred-note";
      wrap.appendChild(rollBar);
      wrap.appendChild(elNote);
      root.appendChild(wrap);
      this.update(app);
    },
    update(app) {
      const top = currentTop(app);
      const chain = app.state.chain ?? 0;
      const T = app.state.temp ?? 1.0;
      const { probs, tail: tailProb } =
        distWithTail(top, T, app.data.model?.vocab_size);
      const rows = top.map((d, i) => ({ text: d.text, prob: probs[i] }))
        .slice(0, TOP_SHOWN);
      const max = Math.max(...rows.map((r) => r.prob), 1e-6);

      elTempLabel.textContent = `Temperatura ${T.toFixed(2)}`;
      elTempFlavor.textContent =
        T <= 0.5 ? "fredda · prevedibile" : T >= 1.3 ? "calda · creativa" : "neutra";
      tempSlider.value = T;   // keep the slider in sync when scroll drives

      elBars.innerHTML = "";
      rows.forEach((r, i) => {
        const row = document.createElement("div");
        row.className = "pred-row" + (i === 0 ? " winner" : "") +
          (lastRoll && lastRoll.i === i ? " sampled" : "");
        row.innerHTML =
          `<span class="pred-word">${escapeHtml(displayToken(r.text))}</span>` +
          `<span class="pred-track"><span class="pred-fill" style="width:${(r.prob / max) * 100}%"></span></span>` +
          `<span class="pred-pct">${(r.prob * 100).toFixed(1)}%</span>`;
        elBars.appendChild(row);
      });

      // the long tail, made visible: every word outside the stored top list,
      // added up. Its bar is capped at full width — the label carries the truth.
      if (tailProb != null) {
        const nOther = (app.data.model?.vocab_size ?? 0) - top.length;
        const row = document.createElement("div");
        row.className = "pred-row other" +
          (lastRoll && lastRoll.i === -1 ? " sampled" : "");
        row.innerHTML =
          `<span class="pred-word">tutto il resto</span>` +
          `<span class="pred-track"><span class="pred-fill" style="width:${Math.min(tailProb / max, 1) * 100}%"></span></span>` +
          `<span class="pred-pct">${(tailProb * 100).toFixed(1)}%</span>`;
        row.title = nOther > 0
          ? `le altre ${nOther.toLocaleString()} parole del vocabolario, in totale`
          : "tutte le parole fuori dall'elenco salvato, in totale";
        elBars.appendChild(row);
      }

      // loop button: label carries the word it will add; exhausted = the moral
      if (growBtn) {
        const contLen = app.data.continuation.length;
        const done = chain >= contLen;
        growBtn.disabled = done;
        growBtn.textContent = done
          ? "…e così via, per sempre"
          : `Aggiungi “${displayToken(top[0].text)}” e prevedi di nuovo →`;
        elChain.textContent = chain > 0
          ? `${chain} ${chain > 1 ? "parole aggiunte" : "parola aggiunta"}`
          : "";
        resetBtn.style.display = chain > 0 ? "" : "none";
      }

      // recent rolls, newest first — at low T they repeat, at high T they wander
      elRolls.innerHTML = "";
      rolls.forEach((roll, j) => {
        const chip = document.createElement("span");
        chip.className = "roll-chip" + (j === 0 ? " latest" : "") +
          (roll.tail ? " tail" : "");
        chip.textContent = displayToken(roll.text);
        chip.title = roll.tail
          ? `finita fuori dalla top 50 — una delle migliaia di parole ` +
            `improbabili (${(roll.prob * 100).toFixed(1)}% di probabilità combinata al lancio)`
          : `${(roll.prob * 100).toFixed(1)}% di probabilità al lancio`;
        elRolls.appendChild(chip);
      });

      // per-example classroom aside (e.g. "why does 'the' beat 'Paris'?") —
      // it explains the BASE prediction, so hide it once the sentence has grown
      const showNote = app.data.note && chain === 0;
      elNote.textContent = showNote ? "💡 " + app.data.note : "";
      elNote.style.display = showNote ? "" : "none";
    },
    onEnter(step, app) {
      // un-reveal when the reader scrolls back up past the reveal step
      app.set({ revealed: step === "reveal" ? currentTop(app)[0].text : null });
    },
    onProgress(step, p, app) {
      if (step === "temperature") app.set({ temp: T_MIN + p * (T_MAX - T_MIN) });
    },
  };
}
