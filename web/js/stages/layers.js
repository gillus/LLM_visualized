// Stage 3 — Layers / logit lens (the centerpiece). Scroll progress drives the
// layer index. Two linked views update together:
//   • left  — a mini "meaning map" (same PCA space as stage 2) where the selected
//             token's hidden state drifts across layers along its trajectory.
//   • right — the top-k guesses, re-ranking from nonsense (early) to the
//             confident answer (late layers).
import { lerp, escapeHtml, displayToken } from "../decode.js";

export function createLayersStage() {
  let elLabel, elBars, mapBox, svg, x, y, hasTraj, toggleEl, slider;

  function setupMap(app, root) {
    const data = app.data;
    hasTraj = Array.isArray(data.trajectory) && data.trajectory.length > 0;
    if (!hasTraj) return;

    const W = mapBox.clientWidth || 300;
    const H = mapBox.clientHeight || 320;
    const allTraj = data.trajectory.flat();
    const pts = [
      ...data.embedding.vocab_cloud.map((d) => d.xy),
      ...data.embedding.tokens.map((d) => d.xy),
      ...allTraj,
    ];
    const pad = 20;
    x = d3.scaleLinear().domain(d3.extent(pts, (d) => d[0])).range([pad, W - pad]);
    y = d3.scaleLinear().domain(d3.extent(pts, (d) => d[1])).range([H - pad, pad]);

    svg = d3.select(mapBox).append("svg").attr("class", "map-svg")
      .attr("viewBox", `0 0 ${W} ${H}`);

    // faint background cloud (dots only — labels would clutter the small box)
    svg.append("g").selectAll("circle").data(data.embedding.vocab_cloud).join("circle")
      .attr("cx", (d) => x(d.xy[0])).attr("cy", (d) => y(d.xy[1]))
      .attr("r", 2).attr("class", "map-cloud-dot");

    // groups drawn on top, refreshed in update()
    svg.append("path").attr("class", "traj-full");      // whole path (faint)
    svg.append("path").attr("class", "traj-done");      // traversed so far (bold)
    svg.append("g").attr("class", "map-tokens");        // start/end + other tokens
    svg.append("circle").attr("class", "traj-marker").attr("r", 7);
    svg.append("text").attr("class", "map-caption")
      .attr("x", 10).attr("y", H - 8).text("dove il token “sta pensando”");
  }

  function line(points) {
    return points.map((p, i) => `${i ? "L" : "M"} ${x(p[0])} ${y(p[1])}`).join(" ");
  }

  function updateMap(app) {
    if (!hasTraj) return;
    const pos = app.state.pos;
    const path = app.data.trajectory[pos];
    const nL = path.length;
    const frac = Math.max(0, Math.min(app.state.layerFrac ?? 0, nL - 1));
    const lo = Math.floor(frac), hi = Math.min(lo + 1, nL - 1), t = frac - lo;
    const cx = lerp(path[lo][0], path[hi][0], t);
    const cy = lerp(path[lo][1], path[hi][1], t);
    const color = app.colorForPos(pos);

    svg.select(".traj-full").attr("d", line(path));
    // traversed portion: points up to lo, plus the interpolated current point
    svg.select(".traj-done").attr("d", line([...path.slice(0, lo + 1), [cx, cy]]))
      .attr("stroke", color);
    svg.select(".traj-marker").attr("cx", x(cx)).attr("cy", y(cy)).attr("fill", color);

    // token anchors: highlight the selected token's start; show others faintly
    const sel = svg.select(".map-tokens").selectAll(".map-tok")
      .data(app.data.embedding.tokens, (d) => d.pos);
    const enter = sel.enter().append("g").attr("class", "map-tok");
    enter.append("circle");
    enter.append("text");
    enter.merge(sel)
      .attr("transform", (d) => `translate(${x(d.xy[0])},${y(d.xy[1])})`)
      .each(function (d) {
        const g = d3.select(this), active = d.pos === pos;
        g.select("circle").attr("r", active ? 5 : 3)
          .attr("class", active ? "tok-start active" : "tok-start");
        g.select("text").attr("x", 7).attr("y", 3)
          .attr("class", "map-tok-label")
          .attr("opacity", active ? 1 : 0)
          .text(app.data.tokens[d.pos].text);
      });
  }

  return {
    id: "layers",
    mount(root, app) {
      root.innerHTML = "";
      const wrap = document.createElement("div");
      wrap.className = "layers-stage";

      const head = document.createElement("div");
      head.className = "layer-head";
      elLabel = document.createElement("div");
      elLabel.className = "layer-label";
      head.appendChild(elLabel);

      // Plain/Tuned lens toggle — only if this example carries a tuned lens.
      toggleEl = null;
      if (app.data.logit_lens_tuned) {
        toggleEl = document.createElement("div");
        toggleEl.className = "lens-toggle";
        toggleEl.innerHTML =
          `<button data-m="tuned" title="Un piccolo traduttore per ogni strato, allenato per ripulire la lettura">Affinata</button>` +
          `<button data-m="plain" title="L'output del modello applicato direttamente a ogni strato (più confuso a metà rete)">Semplice</button>`;
        toggleEl.querySelectorAll("button").forEach((b) => {
          b.onclick = () => app.set({ lensMode: b.dataset.m });
        });
        head.appendChild(toggleEl);
      }
      wrap.appendChild(head);

      // manual layer slider — the non-scroll path (projectors, reduced motion)
      const ctl = document.createElement("div");
      ctl.className = "ctl-row";
      const lbl = document.createElement("span");
      lbl.className = "ctl-lbl";
      lbl.textContent = "strato";
      slider = document.createElement("input");
      slider.type = "range";
      slider.min = 0;
      slider.max = app.data.logit_lens[0].length - 1;
      slider.step = 1;
      slider.setAttribute("aria-label", "strato");
      slider.oninput = () =>
        app.set({ layer: +slider.value, layerFrac: +slider.value });
      ctl.appendChild(lbl);
      ctl.appendChild(slider);
      wrap.appendChild(ctl);

      const body = document.createElement("div");
      body.className = "layers-body";
      mapBox = document.createElement("div");
      mapBox.className = "map-box";
      elBars = document.createElement("div");
      elBars.className = "lens-bars";
      body.appendChild(mapBox);
      body.appendChild(elBars);
      wrap.appendChild(body);
      root.appendChild(wrap);
      setupMap(app, root);
      this.update(app);
    },
    update(app) {
      const pos = app.state.pos;
      const useTuned = app.state.lensMode === "tuned" && app.data.logit_lens_tuned;
      const lensArr = useTuned ? app.data.logit_lens_tuned : app.data.logit_lens;
      const lensForPos = lensArr[pos];
      const nLayers = lensForPos.length; // L+1 (layer 0 = embeddings)
      const layer = Math.min(app.state.layer ?? 0, nLayers - 1);
      const guesses = lensForPos[layer];
      const token = app.data.tokens[pos];

      if (toggleEl) {
        toggleEl.querySelectorAll("button").forEach((b) =>
          b.classList.toggle("active", b.dataset.m === (useTuned ? "tuned" : "plain")));
      }
      slider.value = layer;   // keep the slider in sync when scroll drives

      const isLast = pos === app.data.tokens.length - 1;
      // index 0 is the embeddings, before any layer has run — don't call it a layer
      const layerName = layer === 0 ? "Inizio" : `Strato ${layer}`;
      elLabel.innerHTML =
        `<span class="layer-num">${layerName}<span class="layer-max"> / ${nLayers - 1}</span></span>` +
        `<span class="layer-sub">${layer === 0 ? "solo gli embedding · " : ""}` +
        `<span class="aa-detail">${useTuned ? "lente affinata · " : "lente semplice · "}</span>` +
        `previsione dopo “${token.text}”` +
        (isLast ? " (la vera parola successiva)" : "") + "</span>";

      const max = Math.max(...guesses.map((g) => g.prob), 1e-6);
      elBars.innerHTML = "";
      guesses.forEach((gss) => {
        const row = document.createElement("div");
        row.className = "lens-row";
        row.innerHTML =
          `<span class="lens-word">${escapeHtml(displayToken(gss.text))}</span>` +
          `<span class="lens-track"><span class="lens-fill" style="width:${(gss.prob / max) * 100}%"></span></span>` +
          `<span class="lens-pct">${(gss.prob * 100).toFixed(1)}%</span>`;
        elBars.appendChild(row);
      });

      updateMap(app);
    },
    onEnter(step, app) {
      // the embedding stage walks the selection to a content word; walk it
      // back to the last token here — that's where the real prediction
      // happens. A learner's own click (posPinned) always wins.
      const last = app.data.tokens.length - 1;
      if (step === "intro" && !app.state.posPinned && app.state.pos !== last) {
        app.set({ pos: last });
      }
    },
    onProgress(step, p, app) {
      if (step !== "scrub") return;   // don't let the intro step fight the slider
      const n = app.data.logit_lens[app.state.pos].length;
      const frac = p * (n - 1);
      // layer (int) drives the bars; layerFrac drives the smooth map marker
      app.set({ layer: Math.max(0, Math.min(Math.round(frac), n - 1)), layerFrac: frac });
    },
  };
}
