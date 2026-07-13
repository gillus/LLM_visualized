// Stage 3 — Layers / logit lens (the centerpiece). Scroll progress drives the
// layer index. Two linked views update together:
//   • left  — a mini "meaning map" (same PCA space as stage 2) where the selected
//             token's hidden state drifts across layers along its trajectory.
//             Thin "listening lines" show which context words the active block
//             is attending to (head-averaged, precomputed attention).
//   • right — the top-k guesses, re-ranking from nonsense (early) to the
//             confident answer (late layers), with ▲▼ deltas vs the previous block.
// A rail of identical block chips replaces the old slider: same shape repeated
// N times — the visual argument that it's one kind of block applied over and over.
import { lerp, escapeHtml, displayToken, decodeAttention } from "../decode.js";

export function createLayersStage() {
  let elLabel, elBars, mapBox, svg, x, y, hasTraj, toggleEl, chips, tokXY;
  let attAvg = null;   // [layer][q][k] attention averaged over heads
  let lastLayer = -1;  // for the "block fired" pulse
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

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
    tokXY = new Map(data.embedding.tokens.map((t) => [t.pos, t.xy]));

    svg = d3.select(mapBox).append("svg").attr("class", "map-svg")
      .attr("viewBox", `0 0 ${W} ${H}`);

    // faint background cloud (dots only — labels would clutter the small box)
    svg.append("g").selectAll("circle").data(data.embedding.vocab_cloud).join("circle")
      .attr("cx", (d) => x(d.xy[0])).attr("cy", (d) => y(d.xy[1]))
      .attr("r", 2).attr("class", "map-cloud-dot");

    // groups drawn on top, refreshed in update()
    svg.append("g").attr("class", "attn-lines");        // who the block listens to
    svg.append("path").attr("class", "traj-full");      // whole path (faint)
    svg.append("path").attr("class", "traj-done");      // traversed so far (bold)
    svg.append("g").attr("class", "map-tokens");        // start/end + other tokens
    svg.append("circle").attr("class", "traj-marker").attr("r", 7);
    svg.append("text").attr("class", "map-caption")
      .attr("x", 10).attr("y", H - 8).text("dove il token “sta pensando”");
    svg.append("text").attr("class", "map-caption attn-caption")
      .attr("x", 10).attr("y", 16).attr("opacity", 0)
      .text("linee: chi sta ascoltando questo blocco");
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

    // "listening lines": at each block, which context words is it reading?
    // Uses the head-averaged attention of the block being traversed. Like
    // arch.js, position-0 attention is treated as a "sink" and folded into
    // self (not drawn), so it doesn't drown the real signal; self isn't drawn
    // either — the marker already IS the token.
    let lines = [];
    const block = Math.ceil(frac);         // block being traversed (1..L)
    if (attAvg && block >= 1 && block <= attAvg.length && pos > 1) {
      const row = attAvg[block - 1][pos];
      const cand = [];
      for (let k = 1; k < pos; k++) if (row[k] >= 0.03) cand.push([k, row[k]]);
      cand.sort((a, b) => b[1] - a[1]);
      lines = cand.slice(0, 3);
    }
    const maxW = lines.length ? lines[0][1] : 1;
    const listened = new Map(lines);

    svg.select(".attn-lines").selectAll("line")
      .data(lines, (d) => d[0])
      .join("line")
      .attr("class", "attn-line")
      .attr("x1", (d) => x(tokXY.get(d[0])[0]))
      .attr("y1", (d) => y(tokXY.get(d[0])[1]))
      .attr("x2", x(cx)).attr("y2", y(cy))
      .attr("stroke", (d) => app.colorForPos(d[0]))
      .attr("stroke-width", (d) => 1 + 1.8 * (d[1] / maxW))
      .attr("opacity", (d) => 0.2 + 0.6 * (d[1] / maxW));
    svg.select(".attn-caption").attr("opacity", lines.length ? 1 : 0);

    svg.select(".traj-full").attr("d", line(path));
    // traversed portion: points up to lo, plus the interpolated current point
    svg.select(".traj-done").attr("d", line([...path.slice(0, lo + 1), [cx, cy]]))
      .attr("stroke", color);
    svg.select(".traj-marker").attr("cx", x(cx)).attr("cy", y(cy)).attr("fill", color);

    // token anchors: highlight the selected token's start; label the words the
    // block is currently listening to; show the rest faintly
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
          .attr("opacity", active ? 1 : listened.has(d.pos) ? 0.85 : 0)
          .text(app.data.tokens[d.pos].text);
      });
  }

  return {
    id: "layers",
    mount(root, app) {
      root.innerHTML = "";
      lastLayer = -1;
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

      // block rail — the manual, non-scroll control (projectors, reduced
      // motion): one chip per block, all drawn identical on purpose.
      const ctl = document.createElement("div");
      ctl.className = "ctl-row";
      const lbl = document.createElement("span");
      lbl.className = "ctl-lbl";
      lbl.textContent = "blocchi";
      const rail = document.createElement("div");
      rail.className = "block-rail";
      rail.setAttribute("role", "group");
      rail.setAttribute("aria-label", "scegli il blocco");
      const nL = app.data.logit_lens[0].length; // L+1 (index 0 = embeddings)
      chips = [];
      for (let i = 0; i < nL; i++) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "blk" + (i === 0 ? " blk-start" : "");
        const name = i === 0 ? "Inizio — solo gli embedding" : `Blocco ${i}`;
        b.title = name;
        b.setAttribute("aria-label", name);
        b.onclick = () => app.set({ layer: i, layerFrac: i });
        rail.appendChild(b);
        chips.push(b);
      }
      ctl.appendChild(lbl);
      ctl.appendChild(rail);
      wrap.appendChild(ctl);

      // head-averaged attention per layer, decoded once per example — at
      // scroll time we only read a row out of it
      attAvg = null;
      const att = app.data.attention;
      if (att?.weights?.length) {
        const N = app.data.tokens.length;
        const dense = decodeAttention(att, N);
        attAvg = dense.map((heads) => {
          const H = heads.length;
          const out = [];
          for (let q = 0; q < N; q++) {
            const row = new Array(N).fill(0);
            for (let h = 0; h < H; h++)
              for (let k = 0; k <= q; k++) row[k] += heads[h][q][k];
            for (let k = 0; k < N; k++) row[k] /= H;
            out.push(row);
          }
          return out;
        });
      }

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

      // rail state + "the block fired" pulse when we cross into a new block
      chips.forEach((c, i) => {
        c.classList.toggle("active", i === layer);
        c.classList.toggle("done", i < layer);
      });
      if (layer !== lastLayer) {
        if (lastLayer !== -1 && !reducedMotion.matches) {
          const chip = chips[layer];
          if (chip && layer > 0) {
            chip.classList.remove("fired");
            void chip.offsetWidth;   // restart the one-shot animation
            chip.classList.add("fired");
            elBars.classList.remove("refreshed");
            void elBars.offsetWidth;
            elBars.classList.add("refreshed");
          }
        }
        lastLayer = layer;
      }

      const isLast = pos === app.data.tokens.length - 1;
      // index 0 is the embeddings, before any block has run — don't call it a block
      const layerName = layer === 0 ? "Inizio" : `Blocco ${layer}`;
      const layerMax = layer === 0 ? "" : `<span class="layer-max"> di ${nLayers - 1}</span>`;
      elLabel.innerHTML =
        `<span class="layer-num">${layerName}${layerMax}</span>` +
        `<span class="layer-sub">${layer === 0 ? "solo gli embedding, nessun blocco ancora · " : ""}` +
        `<span class="aa-detail">${useTuned ? "lente affinata · " : "lente semplice · "}</span>` +
        `previsione dopo “${token.text}”` +
        (isLast ? " (la vera parola successiva)" : "") + "</span>";

      // ▲▼ vs the previous block: did this block push a word up or down?
      const prev = layer > 0
        ? new Map(lensForPos[layer - 1].map((g) => [g.text, g.prob]))
        : null;
      const max = Math.max(...guesses.map((g) => g.prob), 1e-6);
      elBars.innerHTML = "";
      guesses.forEach((gss) => {
        let delta = "";
        if (prev) {
          const p0 = prev.get(gss.text);
          if (p0 === undefined)
            delta = `<span class="lens-delta up" title="entrata in classifica dopo questo blocco">▲</span>`;
          else if (gss.prob > p0 * 1.2 + 1e-4)
            delta = `<span class="lens-delta up" title="salita dopo questo blocco">▲</span>`;
          else if (gss.prob < p0 / 1.2 - 1e-4)
            delta = `<span class="lens-delta down" title="scesa dopo questo blocco">▼</span>`;
          else delta = `<span class="lens-delta same">·</span>`;
        } else {
          delta = `<span class="lens-delta"></span>`;
        }
        const row = document.createElement("div");
        row.className = "lens-row";
        row.innerHTML =
          `<span class="lens-word">${escapeHtml(displayToken(gss.text))}</span>` +
          `<span class="lens-track"><span class="lens-fill" style="width:${(gss.prob / max) * 100}%"></span></span>` +
          `<span class="lens-pct">${(gss.prob * 100).toFixed(1)}%</span>` +
          delta;
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
      if (step !== "scrub") return;   // don't let the intro step fight the rail
      const n = app.data.logit_lens[app.state.pos].length;
      const frac = p * (n - 1);
      // layer (int) drives the bars; layerFrac drives the smooth map marker
      app.set({ layer: Math.max(0, Math.min(Math.round(frac), n - 1)), layerFrac: frac });
    },
  };
}
