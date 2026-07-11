// Stage 1 — Tokenization. Step 0 shows the prompt as one unbroken sentence;
// step 1 visibly splits it apart into subword chips; step 2 reveals each
// token's numeric id. Chips are built once in mount() and restyled via CSS
// classes in update() so the split actually animates.
export function createTokensStage() {
  let elRow, chips;

  return {
    id: "tokens",
    mount(root, app) {
      root.innerHTML = "";
      const wrap = document.createElement("div");
      wrap.className = "tok-stage";
      elRow = document.createElement("div");
      elRow.className = "tok-chips";
      chips = app.data.tokens.map((t) => {
        const chip = document.createElement("div");
        chip.className = "tok-chip" + (t.lead_space ? " lead" : "");
        chip.style.setProperty("--c", app.colorForPos(t.pos));
        const txt = document.createElement("span");
        txt.className = "tok-text";
        const id = document.createElement("span");
        id.className = "tok-id";
        id.textContent = "#" + t.id;
        chip.appendChild(txt);
        chip.appendChild(id);
        elRow.appendChild(chip);
        return { chip, txt, token: t };
      });
      wrap.appendChild(elRow);
      root.appendChild(wrap);
      this.update(app);
    },
    update(app) {
      const reveal = app.state.tokenReveal ?? 0; // 0 sentence, 1 chips, 2 ids
      elRow.classList.toggle("split", reveal >= 1);
      elRow.classList.toggle("ids", reveal >= 2);
      chips.forEach(({ chip, txt, token }) => {
        chip.classList.toggle("split", reveal >= 1);
        // once split, make the swallowed leading space visible
        txt.textContent = (reveal >= 1 && token.lead_space ? "␣" : "") + token.text;
      });
    },
    onEnter(step, app) {
      const map = { intro: 0, split: 1, ids: 2 };
      app.set({ tokenReveal: map[step] ?? 0 });
    },
  };
}
