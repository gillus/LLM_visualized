// Stage 4 — Attention. For a selected query token, draw arcs back to the
// earlier tokens it "looks at" (thicker = stronger). An advanced step reveals
// the full causal heatmap grid. Layer/head are set by scroll steps.
export function createAttentionStage() {
  let svg, W, H, layerSlider, headSlider, layerVal, headVal;

  return {
    id: "attention",
    mount(root, app) {
      root.innerHTML = "";

      // manual layer/head picker — the non-scroll path (projectors, reduced motion)
      const ctl = document.createElement("div");
      ctl.className = "ctl-row";
      const mk = (name, max, onset) => {
        const lbl = document.createElement("span");
        lbl.className = "ctl-lbl";
        lbl.textContent = name;
        const s = document.createElement("input");
        s.type = "range";
        s.min = 0;
        s.max = max;
        s.step = 1;
        s.setAttribute("aria-label", name);
        s.oninput = () => onset(+s.value);
        const val = document.createElement("span");
        val.className = "ctl-val";
        ctl.appendChild(lbl);
        ctl.appendChild(s);
        ctl.appendChild(val);
        return [s, val];
      };
      [layerSlider, layerVal] = mk("strato", app.attn.length - 1,
        (v) => app.set({ attnLayer: v }));
      [headSlider, headVal] = mk("testa", app.attn[0].length - 1,
        (v) => app.set({ head: v }));
      root.appendChild(ctl);

      const box = document.createElement("div");
      box.className = "attn-box";
      root.appendChild(box);
      W = box.clientWidth || root.clientWidth || 520;
      H = box.clientHeight || (root.clientHeight || 420) - 40;
      svg = d3.select(box).append("svg").attr("class", "attn-svg")
        .attr("viewBox", `0 0 ${W} ${H}`);
      this.update(app);
    },
    update(app) {
      const N = app.data.tokens.length;
      const layer = Math.min(app.state.attnLayer ?? 0, app.attn.length - 1);
      const head = Math.min(app.state.head ?? 0, app.attn[0].length - 1);
      const query = app.state.pos;
      const mat = app.attn[layer][head]; // dense [q][k]
      // keep the pickers in sync when scroll steps drive the state
      layerSlider.value = layer;
      headSlider.value = head;
      layerVal.textContent = `S${layer}`;
      headVal.textContent = `T${head}`;
      svg.selectAll("*").remove();

      // when this layer/head is a curated "spot", caption what it shows —
      // but only if the caption still applies to the selected query token
      const spot = (app.data.attention.spots || []).find(
        (s) => s.layer === layer && s.head === head &&
               (s.focus == null || query === N - 1));
      svg.append("text").attr("class", "attn-title").attr("x", 16).attr("y", 24)
        .text(`Strato ${layer} · Testa ${head} — ` +
          (spot ? spot.why : `chi guarda “${app.data.tokens[query].text}”?`));

      if (app.state.showGrid) {
        drawGrid(svg, app, mat, N, W, H);
      } else {
        drawArcs(svg, app, mat, query, N, W, H);
      }
    },
    onEnter(step, app) {
      // steps encode "spot<n>" (curated heads from the data), legacy
      // "L<layer>H<head>", or "grid"/"arcs"
      if (step === "grid") return app.set({ showGrid: true });
      if (step === "arcs") return app.set({ showGrid: false });
      const sm = /^spot(\d+)$/.exec(step);
      if (sm) {
        const s = (app.data.attention.spots || [])[+sm[1]] ?? { layer: +sm[1], head: 0 };
        return app.set({ attnLayer: s.layer, head: s.head, showGrid: false });
      }
      const m = /^L(\d+)H(\d+)$/.exec(step);
      if (m) app.set({ attnLayer: +m[1], head: +m[2], showGrid: false });
    },
  };
}

function tokenXs(app, N, W) {
  const pad = 40, span = W - 2 * pad;
  return app.data.tokens.map((_, i) => pad + (N === 1 ? span / 2 : (span * i) / (N - 1)));
}

function drawArcs(svg, app, mat, query, N, W, H) {
  const xs = tokenXs(app, N, W);
  const baseY = H * 0.62;
  // arcs from query back to each key
  const row = mat[query];
  const max = Math.max(...row.slice(0, query + 1), 1e-6);
  for (let k = 0; k <= query; k++) {
    const w = row[k] / max;
    if (w < 0.02) continue;
    const x0 = xs[query], x1 = xs[k], r = Math.abs(x0 - x1) / 2;
    const path = k === query
      ? `M ${x0} ${baseY} m -10 0 a 10 18 0 1 0 20 0 a 10 18 0 1 0 -20 0`
      : `M ${x0} ${baseY} A ${r} ${r * 0.7} 0 0 1 ${x1} ${baseY}`;
    svg.append("path").attr("d", path).attr("class", "attn-arc")
      .attr("stroke", app.colorForPos(query))
      .attr("stroke-width", 1 + w * 9).attr("opacity", 0.25 + w * 0.6);
  }
  // token chips
  app.data.tokens.forEach((t, i) => {
    const gx = xs[i];
    svg.append("text").attr("x", gx).attr("y", baseY + 26)
      .attr("class", "attn-tok" + (i === query ? " query" : ""))
      .attr("text-anchor", "middle").text(t.text)
      .attr("fill", i === query ? app.colorForPos(query) : "#6f6754");
  });
}

function drawGrid(svg, app, mat, N, W, H) {
  const pad = 70, size = Math.min(W, H) - pad - 30;
  const cell = size / N, ox = pad, oy = 50;
  // warm "heat" ramp: paper → burnt orange (matches the field-guide theme)
  const color = d3.scaleSequential(d3.interpolateRgb("#faf5e8", "#b3410f")).domain([0, 1]);
  for (let q = 0; q < N; q++) {
    for (let k = 0; k <= q; k++) {
      svg.append("rect").attr("x", ox + k * cell).attr("y", oy + q * cell)
        .attr("width", cell - 1).attr("height", cell - 1)
        .attr("fill", color(mat[q][k]));
    }
  }
  app.data.tokens.forEach((t, i) => {
    svg.append("text").attr("x", ox - 6).attr("y", oy + i * cell + cell / 2 + 4)
      .attr("class", "grid-lbl").attr("text-anchor", "end").text(t.text);
    svg.append("text").attr("x", ox + i * cell + cell / 2).attr("y", oy - 6)
      .attr("class", "grid-lbl").attr("text-anchor", "middle").text(t.text);
  });
  svg.append("text").attr("x", ox).attr("y", oy + size + 22)
    .attr("class", "grid-axis").text("riga = query (chi chiede) · colonna = key (chi viene guardato)");
}
