// Stage 4 — Prediction. The final next-token distribution as a bar chart,
// plus a temperature control that SNAPS to the precomputed bins in
// data.temp_chains (see SCHEMA.md). For the current bin the panel plays a
// continuation the model actually sampled offline at that temperature; the
// "Un'altra!" button cycles the stored variants — cold = the same sentence
// every time, hot = a different (and sometimes strange) one per try. The
// final scroll step reveals the chain's first word in the sticky token strip.
import { escapeHtml, displayToken, progressToIndex } from "../decode.js";

const TOP_SHOWN = 8;
const CHIP_MS = 180;   // one chain chip appears per tick
// files without temp_chains (older data): the slider keeps these detents and
// the chain panel simply stays hidden
const FALLBACK_BINS = [{ t: 0 }, { t: 0.6 }, { t: 0.9 }, { t: 1.3 }, { t: 1.8 }];
const REDUCED = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

const binsOf = (app) => app.data.temp_chains?.bins ?? FALLBACK_BINS;

function currentBin(app) {
  const bins = binsOf(app);
  return bins[Math.min(app.state.tempBin ?? 0, bins.length - 1)];
}

// the sampled chain currently on stage (null when the data has no chains)
function currentVariant(app) {
  const vs = currentBin(app).variants;
  return vs?.length ? vs[(app.state.variant ?? 0) % vs.length] : null;
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
  const hasTail = nOther > 0 && top[0].prob > 0 && mass < 0.999;
  let wTail = 0;
  if (hasTail) {
    // per-word tail logit, recovered from softmax(l₀) = p₀ at T = 1
    const tailLogit = top[0].logit - Math.log(top[0].prob) +
      Math.log((1 - mass) / nOther);
    wTail = nOther * Math.exp((tailLogit - m) / t);
  }
  const sum = w.reduce((a, b) => a + b, 0) + wTail;
  // tail is 0 (not null) when it merely froze out at cold T — the row should
  // stay visible at 0.0% instead of popping in and out of the layout
  return { probs: w.map((v) => v / sum), tail: hasTail ? wTail / sum : null };
}

export function createPredictionStage() {
  let elTempLabel, elTempFlavor, tempSlider, elBars, elNote;
  let elChainPanel, elChainStrip, elChainMeta, variantBtn;
  let chainTimer = null;
  let chainKey = null;      // which (example, bin, variant) the strip is showing
  let revealActive = false; // true while the "variants" scroll step is current

  function stopChainTimer() {
    if (chainTimer) { clearInterval(chainTimer); chainTimer = null; }
  }

  function chipFor(app, t, i) {
    const chip = document.createElement("span");
    chip.className = "chain-chip" + (t.lead_space ? " lead" : "");
    chip.style.setProperty("--c",
      app.colorForPos(app.data.tokens.length + i));
    chip.textContent = displayToken(t.text);
    return chip;
  }

  // (re)play the current variant in the strip: instant under reduced motion,
  // otherwise one chip per tick — plain DOM appends, no app.set per tick
  function playChain(app, v) {
    stopChainTimer();
    elChainStrip.innerHTML = "";
    if (REDUCED) {
      v.forEach((t, i) => elChainStrip.appendChild(chipFor(app, t, i)));
      return;
    }
    let i = 0;
    chainTimer = setInterval(() => {
      if (i >= v.length) { stopChainTimer(); return; }
      elChainStrip.appendChild(chipFor(app, v[i], i));
      i++;
    }, CHIP_MS);
  }

  return {
    id: "prediction",
    mount(root, app) {
      root.innerHTML = "";
      stopChainTimer();
      chainKey = null;
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
      tempSlider.min = 0;
      tempSlider.max = binsOf(app).length - 1;
      tempSlider.step = 1;
      tempSlider.setAttribute("aria-label", "temperatura");
      tempSlider.oninput = () =>
        app.set({ tempBin: +tempSlider.value, variant: 0 });
      elTempFlavor = document.createElement("span");
      elTempFlavor.className = "temp-flavor";
      elTemp.appendChild(elTempLabel);
      elTemp.appendChild(tempSlider);
      elTemp.appendChild(elTempFlavor);
      elBars = document.createElement("div");
      elBars.className = "pred-bars";
      wrap.appendChild(elTemp);
      wrap.appendChild(elBars);

      // the sampled continuation for the current temperature bin
      elChainPanel = document.createElement("div");
      elChainPanel.className = "pred-chain";
      const lbl = document.createElement("p");
      lbl.className = "chain-lbl";
      lbl.textContent = "Il modello continua la frase:";
      elChainStrip = document.createElement("div");
      elChainStrip.className = "chain-strip";
      const actions = document.createElement("div");
      actions.className = "chain-actions";
      variantBtn = document.createElement("button");
      variantBtn.type = "button";
      variantBtn.className = "variant-btn";
      variantBtn.textContent = "Un'altra!";
      variantBtn.title = "Un altro tentativo alla stessa temperatura";
      variantBtn.onclick = () =>
        app.set({ variant: (app.state.variant ?? 0) + 1 });
      elChainMeta = document.createElement("span");
      elChainMeta.className = "chain-meta";
      actions.appendChild(variantBtn);
      actions.appendChild(elChainMeta);
      elChainPanel.appendChild(lbl);
      elChainPanel.appendChild(elChainStrip);
      elChainPanel.appendChild(actions);
      if (!app.data.temp_chains) elChainPanel.style.display = "none";
      wrap.appendChild(elChainPanel);

      elNote = document.createElement("div");
      elNote.className = "pred-note";
      wrap.appendChild(elNote);
      root.appendChild(wrap);
      this.update(app);
    },
    update(app) {
      const bins = binsOf(app);
      const bi = Math.min(app.state.tempBin ?? 0, bins.length - 1);
      const T = bins[bi].t;
      const top = app.data.prediction.top;
      const { probs, tail: tailProb } =
        distWithTail(top, T, app.data.model?.vocab_size);
      const rows = top.map((d, i) => ({ text: d.text, prob: probs[i] }))
        .slice(0, TOP_SHOWN);
      const max = Math.max(...rows.map((r) => r.prob), 1e-6);

      elTempLabel.textContent =
        T === 0 ? "Temperatura 0" : `Temperatura ${T.toFixed(1)}`;
      elTempFlavor.textContent =
        T === 0 ? "ghiacciata · sempre uguale"
        : T <= 0.7 ? "fredda · prevedibile"
        : T >= 1.2 ? "calda · creativa" : "neutra";
      tempSlider.value = bi;   // keep the slider in sync when scroll drives

      elBars.innerHTML = "";
      rows.forEach((r, i) => {
        const row = document.createElement("div");
        row.className = "pred-row" + (i === 0 ? " winner" : "");
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
        row.className = "pred-row other";
        row.innerHTML =
          `<span class="pred-word">tutto il resto</span>` +
          `<span class="pred-track"><span class="pred-fill" style="width:${Math.min(tailProb / max, 1) * 100}%"></span></span>` +
          `<span class="pred-pct">${(tailProb * 100).toFixed(1)}%</span>`;
        row.title = nOther > 0
          ? `le altre ${nOther.toLocaleString()} parole del vocabolario, in totale`
          : "tutte le parole fuori dall'elenco salvato, in totale";
        elBars.appendChild(row);
      }

      // the sampled continuation for this bin — replay only when the
      // (example, bin, variant) on stage actually changed
      const v = currentVariant(app);
      if (v) {
        const nvar = bins[bi].variants.length;
        const vi = (app.state.variant ?? 0) % nvar;
        elChainMeta.textContent = nvar > 1
          ? `tentativo ${vi + 1} di ${nvar} — stessa temperatura, altro tiro`
          : "a questa temperatura esce sempre uguale";
        variantBtn.disabled = nvar <= 1;
        const key = `${app.data.id}|${bi}|${vi}`;
        if (key !== chainKey) {
          chainKey = key;
          playChain(app, v);
        }
        // while the last scroll step is active, the chain's first word fills
        // the "?" slot in the sticky strip (guarded: set → render → equal → stop)
        const want = revealActive ? (v[0]?.text ?? null) : app.state.revealed;
        if (want !== app.state.revealed) app.set({ revealed: want });
      }

      // per-example classroom aside (e.g. "why does 'the' beat 'Paris'?")
      elNote.textContent = app.data.note ? "💡 " + app.data.note : "";
      elNote.style.display = app.data.note ? "" : "none";
    },
    onEnter(step, app) {
      // un-reveal when the reader scrolls back up past the final step
      revealActive = step === "variants";
      app.set({
        revealed: revealActive ? (currentVariant(app)?.[0]?.text ?? null) : null,
      });
    },
    onProgress(step, p, app) {
      if (step !== "temperature") return;
      const bi = progressToIndex(p, binsOf(app).length);
      if (bi !== (app.state.tempBin ?? 0)) app.set({ tempBin: bi, variant: 0 });
    },
  };
}
