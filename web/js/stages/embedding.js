// Stage 2 — Embedding "meaning map". A 2D PCA scatter: background vocabulary
// cloud (gray) + the prompt's tokens (colored). Step 1 draws the selected
// token's nearest neighbors at their own map positions, linked by lines;
// step 2 lights up every prompt token's neighborhood in that token's color.
import { displayToken } from "../decode.js";

export function createEmbeddingStage() {
  let svg, g, x, y, W, H;

  return {
    id: "embedding",
    mount(root, app) {
      root.innerHTML = "";
      W = root.clientWidth || 520;
      H = root.clientHeight || 420;
      svg = d3.select(root).append("svg")
        .attr("class", "emb-svg")
        .attr("viewBox", `0 0 ${W} ${H}`);
      g = svg.append("g");

      // Fixed scales from the extent of ALL points (cloud + tokens + neighbors),
      // so positions are stable across steps and comparable across examples.
      const nbrXY = Object.values(app.data.embedding.neighbors)
        .flat().map((n) => n.xy).filter(Boolean);
      const pts = [
        ...app.data.embedding.vocab_cloud.map((d) => d.xy),
        ...app.data.embedding.tokens.map((d) => d.xy),
        ...nbrXY,
      ];
      const pad = 24;
      x = d3.scaleLinear().domain(d3.extent(pts, (d) => d[0])).range([pad, W - pad]);
      y = d3.scaleLinear().domain(d3.extent(pts, (d) => d[1])).range([H - pad, pad]);

      // background cloud
      g.append("g").attr("class", "cloud")
        .selectAll("text").data(app.data.embedding.vocab_cloud).join("text")
        .attr("x", (d) => x(d.xy[0])).attr("y", (d) => y(d.xy[1]))
        .attr("class", "cloud-word").text((d) => d.text);

      // neighbor links + labels sit above the cloud, below the prompt tokens
      g.append("g").attr("class", "neigh-links");
      g.append("g").attr("class", "neigh-words");
      g.append("g").attr("class", "emb-tokens");
      this.update(app);
    },
    update(app) {
      const step = app.state.embStep ?? 0; // 0 all, 1 neighbors, 2 clusters
      const pos = app.state.pos;
      const nbrData = app.data.embedding.neighbors;
      const hasXY = Object.values(nbrData)[0]?.[0]?.xy != null;

      // which neighborhoods to draw: none / the selected token's / every token's
      const show = step === 1 ? [pos]
        : step >= 2 ? app.data.tokens.map((t) => t.pos) : [];

      const seen = new Set();
      const marks = [];
      if (hasXY) {
        for (const p of show) {
          for (const n of nbrData[String(p)] || []) {
            if (seen.has(n.id)) continue;
            seen.add(n.id);
            marks.push({ id: n.id, text: n.text, xy: n.xy, pos: p, sim: n.sim });
          }
        }
      }

      svg.selectAll(".cloud-word").classed("dim", step >= 1);
      // legacy data without neighbor xy: fall back to highlighting cloud words
      const textMatch = new Set(!hasXY
        ? show.flatMap((p) => (nbrData[String(p)] || []).map((n) => n.text)) : []);
      svg.selectAll(".cloud-word").classed("neighbor", (d) => textMatch.has(d.text));

      const tokXY = new Map(app.data.embedding.tokens.map((d) => [d.pos, d.xy]));
      g.select(".neigh-links").selectAll("line")
        .data(marks, (d) => d.id)
        .join("line")
        .attr("class", "neigh-link")
        .attr("x1", (d) => x(tokXY.get(d.pos)[0]))
        .attr("y1", (d) => y(tokXY.get(d.pos)[1]))
        .attr("x2", (d) => x(d.xy[0])).attr("y2", (d) => y(d.xy[1]))
        .attr("stroke", (d) => app.colorForPos(d.pos));
      g.select(".neigh-words").selectAll("text")
        .data(marks, (d) => d.id)
        .join("text")
        .attr("class", "neigh-word")
        .attr("x", (d) => x(d.xy[0])).attr("y", (d) => y(d.xy[1]))
        .attr("fill", (d) => app.colorForPos(d.pos))
        .each(function (d) {
          const t = d3.select(this).text(displayToken(d.text));
          if (d.sim != null) {   // raw cosine similarity — detail mode only
            t.append("tspan").attr("class", "aa-sim aa-detail")
              .attr("dx", 4).text(d.sim.toFixed(2));
          }
        });

      const sel = g.select(".emb-tokens").selectAll(".emb-tok")
        .data(app.data.embedding.tokens, (d) => d.pos);
      const enter = sel.enter().append("g").attr("class", "emb-tok");
      enter.append("circle");
      enter.append("text");
      enter.merge(sel).attr("transform", (d) => `translate(${x(d.xy[0])},${y(d.xy[1])})`)
        .each(function (d) {
          const node = d3.select(this);
          const token = app.data.tokens[d.pos];
          const active = d.pos === pos;
          node.select("circle")
            .attr("r", active ? 8 : 6)
            .attr("fill", app.colorForPos(d.pos))
            .classed("active", active);
          node.select("text").attr("x", 10).attr("y", 4)
            .attr("class", "emb-label").text(token.text);
        });
    },
    onEnter(step, app) {
      const map = { all: 0, neighbors: 1, clusters: 2 };
      const partial = { embStep: map[step] ?? 0 };
      // walk the selection to the curated content word ("cat", "France") for
      // the neighbors reveal — the last token is always a boring function
      // word. A learner's own click (posPinned) always wins.
      const hero = app.data.embedding.hero;
      if (step === "neighbors" && hero != null && !app.state.posPinned) {
        partial.pos = hero;
      }
      app.set(partial);
    },
  };
}
