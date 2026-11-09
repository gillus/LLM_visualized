// "Come impara una macchina?" — the neural-network lab (neural-net.html).
//
// Division of labour: Python (web/py/nnlab.py, run in the page by PyScript)
// does ALL the maths — training the line and the network, predictions,
// errors. It exposes window.NNPY (see web/py/main.py), whose calls return
// JSON strings. This file only draws those numbers with d3 and wires the
// controls. The house data (web/house_prices.json) is fetched here too, so
// the data section renders before Python has finished starting.
import { showLoadError } from "./aa.js";
import { mountGlossary } from "./glossary.js";

const RM = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (id) => document.getElementById(id);

const LINE_STEPS_PER_RUN = 60;   // one "Allena" press on the line
const LINE_TICK_MS = 70;         // ~14 steps/s: slow enough to watch it wobble
const NN_EPOCHS_PER_RUN = 800;   // one "Allena" press on the network
const NN_FRAME_BUDGET_MS = 28;   // Python time per animation frame

const ARCH_INFO = {
  "1": { name: "1 neurone", params: 6 },
  "2": { name: "2 neuroni", params: 11 },
  "2x2": { name: "2 strati × 2 neuroni", params: 17 },
};

const S = {
  data: null, houses: [], py: null,
  dirData: 0, dirNN: 1, arch: "1",
  line: null, lineHist: [], lineRun: false, lineTimer: null, lineTouched: false,
  nn: null, nnRun: false, nnTarget: 0, nnToken: 0,
  report: {},
  colorByDist: false,
  charts: {},
};

// ---------- formatting ----------
const fmtInt = (v) => Math.round(v).toLocaleString("it-IT");
const fmtEur = (v) => {
  const k = Math.round(v / 1000);
  return k === 0 ? "0 €" : `${k.toLocaleString("it-IT")}.000 €`;
};
const eurAxis = (v) => (v === 0 ? "0" : `${(v / 1000).toLocaleString("it-IT")} mila €`);
function fmtFeat(j, v, withUnit = true) {
  const f = S.data.features[j];
  const s = v.toLocaleString("it-IT", {
    minimumFractionDigits: f.decimals, maximumFractionDigits: f.decimals,
    useGrouping: false,
  });
  return withUnit && f.unit ? `${s} ${f.unit}` : s;
}
const featAxisLabel = (j) => {
  const f = S.data.features[j];
  return f.unit ? `${f.label} (${f.unit})` : f.label;
};
function houseTitle(h) {
  const [m2, km, year] = h.x;
  return `${fmtFeat(0, m2)} · ${fmtFeat(1, km)} dal centro · costruita nel ` +
    `${fmtFeat(2, year)} → ${fmtEur(h.price)}` + (h.test ? " (tenuta da parte)" : "");
}

// ---------- shared chart scaffold ----------
function makeChart(el, { xDomain, yDomain, xLabel, xFmt, label }) {
  el.innerHTML = "";
  const W = Math.max(280, el.clientWidth || 600);
  const H = Math.max(250, Math.round(W * 0.6));
  const narrow = W < 460;
  const m = { t: 14, r: 14, b: 46, l: narrow ? 62 : 78 };
  const svg = d3.select(el).append("svg")
    .attr("class", "nn-svg").attr("role", "img").attr("aria-label", label)
    .attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
  const x = d3.scaleLinear().domain(xDomain).nice().range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain(yDomain).nice().range([H - m.b, m.t]);
  svg.append("g").attr("class", "nn-axis y")
    .attr("transform", `translate(${m.l},0)`)
    .call(d3.axisLeft(y).ticks(5).tickFormat(eurAxis).tickSize(-(W - m.l - m.r)))
    .call((g) => g.select(".domain").remove());
  svg.append("g").attr("class", "nn-axis x")
    .attr("transform", `translate(0,${H - m.b})`)
    .call(d3.axisBottom(x).ticks(narrow ? 4 : 7).tickFormat(xFmt));
  svg.append("text").attr("class", "nn-axis-lbl")
    .attr("x", (m.l + W - m.r) / 2).attr("y", H - 8)
    .attr("text-anchor", "middle").text(xLabel);
  const clip = `clip-${el.id}`;
  svg.append("clipPath").attr("id", clip).append("rect")
    .attr("x", m.l).attr("y", m.t)
    .attr("width", W - m.l - m.r).attr("height", H - m.t - m.b);
  const plot = svg.append("g").attr("clip-path", `url(#${clip})`);
  return { svg, plot, x, y, W, H, m };
}

const priceDomain = () => [0, d3.max(S.houses, (h) => h.price) * 1.05];
const featDomain = (j) => d3.extent(S.houses, (h) => h.x[j]);

function buildChips(host, current, onPick) {
  host.innerHTML = "";
  S.data.features.forEach((f, j) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = f.label;
    b.setAttribute("aria-pressed", String(j === current));
    b.addEventListener("click", () => {
      host.querySelectorAll(".chip").forEach((c, k) =>
        c.setAttribute("aria-pressed", String(k === j)));
      onPick(j);
    });
    host.appendChild(b);
  });
}

// =====================================================================
// ① DATA
// =====================================================================
function buildDataTable() {
  const feats = S.data.features;
  const head = `<thead><tr><th>#</th>${feats.map((f) =>
    `<th>${f.label}${f.unit ? ` <small>(${f.unit})</small>` : ""}</th>`).join("")}` +
    `<th>prezzo</th><th></th></tr></thead>`;
  const rows = S.houses.map((h, i) =>
    `<tr class="${h.test ? "is-test" : ""}"><td>${i + 1}</td>` +
    h.x.map((v, j) => `<td>${fmtFeat(j, v, false)}</td>`).join("") +
    `<td class="num">${fmtEur(h.price)}</td>` +
    `<td>${h.test ? '<span class="tag-test">da parte</span>' : ""}</td></tr>`).join("");
  $("data-table").innerHTML = head + `<tbody>${rows}</tbody>`;
  $("n-train").textContent = S.data.train.length;
  $("n-test").textContent = S.data.test.length;
}

function drawDataChart() {
  const j = S.dirData;
  const c = makeChart($("data-chart"), {
    xDomain: featDomain(j), yDomain: priceDomain(), xLabel: featAxisLabel(j),
    xFmt: (v) => fmtFeat(j, v, false),
    label: `Grafico: prezzo delle case rispetto a ${S.data.features[j].label}`,
  });
  c.plot.selectAll("circle").data(S.houses).join("circle")
    .attr("class", (h) => (h.test ? "dot-test" : "dot-train"))
    .attr("r", 4.5)
    .attr("cx", (h) => c.x(h.x[j])).attr("cy", (h) => c.y(h.price))
    .append("title").text(houseTitle);
}

// =====================================================================
// ② THE LINE
// =====================================================================
const distColor = d3.scaleLinear().domain([0.5, 2.5, 5])
  .range(["#db2777", "#e7a1bd", "#d8ccb0"]).clamp(true);

function buildLineChart() {
  const c = makeChart($("line-chart"), {
    xDomain: featDomain(0), yDomain: priceDomain(), xLabel: featAxisLabel(0),
    xFmt: (v) => fmtFeat(0, v, false),
    label: "Grafico: la linea che prova a indovinare il prezzo dalla superficie",
  });
  c.gErr = c.plot.append("g").attr("class", "err-bars");
  c.gDots = c.plot.append("g");
  c.lineEl = c.plot.append("line").attr("class", "fit-line");
  S.charts.line = c;
  const train = S.houses.filter((h) => !h.test);
  c.gDots.selectAll("circle").data(train).join("circle")
    .attr("class", "dot-train").attr("r", 4.5)
    .attr("cx", (h) => c.x(h.x[0])).attr("cy", (h) => c.y(h.price))
    .append("title").text(houseTitle);
  paintLineDots();
  if (S.line) drawLine();
}

function paintLineDots() {
  const c = S.charts.line;
  if (!c) return;
  c.gDots.selectAll("circle")
    .style("fill", (h) => (S.colorByDist ? distColor(h.x[1]) : null));
  document.querySelector("#line-legend .dist-key").hidden = !S.colorByDist;
}

function drawLine() {
  const c = S.charts.line, L = S.line;
  if (!c || !L) return;
  const pred = (m2) => L.slope * m2 + L.base;
  const [x0, x1] = c.x.domain();
  c.lineEl.attr("x1", c.x(x0)).attr("y1", c.y(pred(x0)))
    .attr("x2", c.x(x1)).attr("y2", c.y(pred(x1)));
  const train = S.houses.filter((h) => !h.test);
  c.gErr.selectAll("line").data(train).join("line")
    .attr("x1", (h) => c.x(h.x[0])).attr("x2", (h) => c.x(h.x[0]))
    .attr("y1", (h) => c.y(h.price)).attr("y2", (h) => c.y(pred(h.x[0])));

  $("line-mae").textContent = fmtEur(L.mae_train);
  $("line-steps").textContent = L.steps;
  $("line-formula").innerHTML =
    `prezzo = <b>${fmtInt(L.slope)} €</b> × m² ${L.base < 0 ? "−" : "+"} <b>${fmtEur(Math.abs(L.base))}</b>`;
  if (document.activeElement !== $("line-slope")) $("line-slope").value = Math.round(L.slope);
  if (document.activeElement !== $("line-base")) $("line-base").value = Math.round(L.base);
  drawSpark($("line-spark"), S.lineHist);
}

function drawSpark(svgEl, hist) {
  const svg = d3.select(svgEl);
  if (hist.length < 2) { svg.selectAll("*").remove(); return; }
  const x = d3.scaleLinear().domain([0, hist.length - 1]).range([1, 199]);
  const y = d3.scaleLinear().domain([0, d3.max(hist)]).range([34, 3]);
  svg.selectAll("path").data([hist]).join("path")
    .attr("d", d3.line().x((_, i) => x(i)).y((v) => y(v)));
}

function applyLine(json) {
  S.line = JSON.parse(json);
  S.lineHist.push(S.line.mae_train);
  if (S.lineHist.length > 240) S.lineHist.shift();
  drawLine();
  if (S.line.steps > 0 || S.lineTouched) {
    S.report.line = { mae_train: S.line.mae_train, mae_test: S.line.mae_test };
    drawReport();
  }
}

function setLineBtn() {
  $("line-train").textContent = S.lineRun ? "❚❚ Pausa" : "▶ Allena";
}
function stopLine() {
  S.lineRun = false;
  clearTimeout(S.lineTimer);
  setLineBtn();
}
function startLine() {
  const target = S.line.steps + LINE_STEPS_PER_RUN;
  if (RM) { applyLine(S.py.line_step(LINE_STEPS_PER_RUN)); return; }
  S.lineRun = true;
  setLineBtn();
  const tick = () => {
    if (!S.lineRun) return;
    applyLine(S.py.line_step(1));
    if (S.line.steps >= target) stopLine();
    else S.lineTimer = setTimeout(tick, LINE_TICK_MS);
  };
  tick();
}

function wireLine() {
  $("line-train").addEventListener("click", () => (S.lineRun ? stopLine() : startLine()));
  $("line-one").addEventListener("click", () => { stopLine(); applyLine(S.py.line_step(1)); });
  $("line-reset").addEventListener("click", () => {
    stopLine();
    S.lineHist = [];
    S.lineTouched = false;
    applyLine(S.py.line_reset());
  });
  const onSlide = () => {
    stopLine();
    S.lineTouched = true;
    applyLine(S.py.line_set(+$("line-slope").value, +$("line-base").value));
  };
  $("line-slope").addEventListener("input", onSlide);
  $("line-base").addEventListener("input", onSlide);
  $("line-color").addEventListener("change", (e) => {
    S.colorByDist = e.target.checked;
    paintLineDots();
  });
}

// =====================================================================
// ③ THE NETWORK
// =====================================================================
function buildNNChart() {
  const j = S.dirNN;
  const c = makeChart($("nn-chart"), {
    xDomain: featDomain(j), yDomain: priceDomain(), xLabel: featAxisLabel(j),
    xFmt: (v) => fmtFeat(j, v, false),
    label: `Grafico: previsioni della rete contro le case mai viste, lungo ${S.data.features[j].label}`,
  });
  c.gRes = c.plot.append("g").attr("class", "res-lines");
  c.gTrue = c.plot.append("g");
  c.gPred = c.plot.append("g");
  c.curve = c.plot.append("path").attr("class", "nn-curve");
  S.charts.nn = c;
  const test = S.houses.filter((h) => h.test);
  c.gTrue.selectAll("circle").data(test).join("circle")
    .attr("class", "dot-test solid").attr("r", 4.5)
    .attr("cx", (h) => c.x(h.x[j])).attr("cy", (h) => c.y(h.price))
    .append("title").text(houseTitle);
  if (S.nn) drawNNChart();
}

function drawNNChart() {
  const c = S.charts.nn, N = S.nn;
  if (!c || !N || N.direction !== S.dirNN) return;
  const j = S.dirNN;
  const test = S.houses.filter((h) => h.test);
  const rows = test.map((h, i) => ({ h, p: N.test_pred[i] }));
  c.gRes.selectAll("line").data(rows).join("line")
    .attr("x1", (d) => c.x(d.h.x[j])).attr("x2", (d) => c.x(d.h.x[j]))
    .attr("y1", (d) => c.y(d.h.price)).attr("y2", (d) => c.y(d.p));
  c.gPred.selectAll("circle").data(rows).join("circle")
    .attr("class", "ring-pred").attr("r", 4.5)
    .attr("cx", (d) => c.x(d.h.x[j])).attr("cy", (d) => c.y(d.p));
  c.curve.attr("d", d3.line().x((v, k) => c.x(N.xs[k])).y((v) => c.y(v))(N.curve));

  const others = S.data.features.map((f, k) => k).filter((k) => k !== j);
  const desc = { 0: (v) => fmtFeat(0, v), 1: (v) => `${fmtFeat(1, v)} dal centro`,
    2: (v) => `costruita nel ${fmtFeat(2, v)}` };
  $("typical-note").textContent =
    `La linea blu è una «casa tipica» (${others.map((k) => desc[k](N.typical[k])).join(", ")}) ` +
    `di cui cambiamo solo ${S.data.features[j].key === "anno" ? "l'" : "la "}` +
    `${S.data.features[j].label}.`;
}

function drawNet() {
  const el = $("net-diagram"), N = S.nn;
  if (!N) return;
  el.innerHTML = "";
  const W = Math.max(280, el.clientWidth || 520);
  const H = Math.max(240, Math.round(W * 0.6));
  const narrow = W < 460;
  const sizes = [N.layers[0].W[0].length, ...N.layers.map((L) => L.W.length)];
  const nCols = sizes.length;
  const inW = narrow ? 84 : 158, outW = narrow ? 64 : 80, boxH = 34, R = narrow ? 22 : 27;
  const colX = (ci) => {
    const left = 8 + inW / 2, right = W - 8 - outW / 2;
    return left + (right - left) * ci / (nCols - 1);
  };
  const rowY = (n, ri) => H * (ri + 1) / (n + 1);
  const nodes = sizes.map((n, ci) => d3.range(n).map((ri) => ({ x: colX(ci), y: rowY(n, ri), ci, ri })));
  const halfW = (ci) => (ci === 0 ? inW / 2 : ci === nCols - 1 ? outW / 2 : R);

  const svg = d3.select(el).append("svg").attr("class", "net-svg")
    .attr("role", "img").attr("aria-label", `Schema della rete: ${ARCH_INFO[N.arch].name}`)
    .attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);

  // edges: one per weight
  const edges = [];
  N.layers.forEach((L, li) => L.W.forEach((row, o) => row.forEach((w, i) =>
    edges.push({ a: nodes[li][i], b: nodes[li + 1][o], w }))));
  svg.append("g").selectAll("line").data(edges).join("line")
    .attr("class", (e) => `edge ${e.w >= 0 ? "pos" : "neg"}`)
    .attr("x1", (e) => e.a.x + halfW(e.a.ci)).attr("y1", (e) => e.a.y)
    .attr("x2", (e) => e.b.x - halfW(e.b.ci)).attr("y2", (e) => e.b.y)
    .attr("stroke-width", (e) => Math.min(1 + Math.abs(e.w) * 2.2, 10))
    .append("title").text((e) => `peso ${e.w.toLocaleString("it-IT", { maximumFractionDigits: 2 })}`);

  // inputs
  const gIn = svg.append("g").selectAll("g").data(nodes[0]).join("g")
    .attr("class", (d) => `node-in ${d.ri === S.dirNN ? "on" : ""}`)
    .attr("transform", (d) => `translate(${d.x},${d.y})`);
  gIn.append("rect").attr("x", -inW / 2).attr("y", -boxH / 2)
    .attr("width", inW).attr("height", boxH).attr("rx", 7);
  gIn.append("text").attr("text-anchor", "middle").attr("dy", "0.35em")
    .text((d) => (narrow ? ["superficie", "distanza", "anno"][d.ri] : S.data.features[d.ri].label));

  // hidden neurons, each with its own S-curve along the chosen direction
  const sx = d3.scaleLinear().domain([0, N.xs.length - 1]).range([-R * 0.66, R * 0.66]);
  const sy = d3.scaleLinear().domain([0, 1]).range([R * 0.5, -R * 0.5]);
  N.hidden.forEach((layerActs, li) => {
    const g = svg.append("g").selectAll("g").data(nodes[li + 1]).join("g")
      .attr("class", "node-hid").attr("transform", (d) => `translate(${d.x},${d.y})`);
    g.append("circle").attr("r", R);
    g.append("line").attr("class", "spark-base")
      .attr("x1", sx(0)).attr("x2", sx(N.xs.length - 1)).attr("y1", sy(0)).attr("y2", sy(0));
    g.append("path").attr("class", "spark")
      .attr("d", (d) => d3.line().x((v, k) => sx(k)).y((v) => sy(v))(layerActs[d.ri]));
    g.append("title").text((d) =>
      `Neurone ${d.ri + 1}${N.hidden.length > 1 ? ` dello strato ${li + 1}` : ""}: ` +
      `quanto si accende (da 0 a 1) cambiando la ${S.data.features[S.dirNN].label}`);
  });

  // output
  const o = nodes[nCols - 1][0];
  const gOut = svg.append("g").attr("class", "node-out").attr("transform", `translate(${o.x},${o.y})`);
  gOut.append("rect").attr("x", -outW / 2).attr("y", -boxH / 2)
    .attr("width", outW).attr("height", boxH).attr("rx", 7);
  gOut.append("text").attr("text-anchor", "middle").attr("dy", "0.35em").text("prezzo");

  const f = S.data.features[S.dirNN];
  $("net-caption").innerHTML =
    `Dentro ogni neurone: quanto <strong>si accende</strong> (da 0 in basso a 1 in alto) ` +
    `se cambi solo ${f.key === "anno" ? "l'" : "la "}<strong>${f.label}</strong> di una casa tipica.`;
}

function applyNN(json, draw = true) {
  S.nn = JSON.parse(json);
  if (draw) { drawNet(); drawNNChart(); }
  $("nn-epochs").textContent = fmtInt(S.nn.epochs);
  $("nn-mae-train").textContent = S.nn.epochs ? fmtEur(S.nn.mae_train) : "–";
  $("nn-mae-test").textContent = S.nn.epochs ? fmtEur(S.nn.mae_test) : "–";
  if (S.nn.epochs > 0) {
    S.report[S.nn.arch] = { mae_train: S.nn.mae_train, mae_test: S.nn.mae_test };
    drawReport();
  }
}

function setNNBtn() {
  $("nn-train").textContent = S.nnRun ? "❚❚ Pausa"
    : S.nn && S.nn.epochs > 0 ? "▶ Allena ancora" : "▶ Allena";
}
function stopNN() { S.nnRun = false; setNNBtn(); }
function startNN() {
  S.nnTarget = S.nn.epochs + NN_EPOCHS_PER_RUN;
  S.nnRun = true;
  setNNBtn();
  const token = ++S.nnToken;   // a quick pause→resume must not leave two loops running
  let n = 3;
  // Train in small chunks so the page never freezes. With reduced motion the
  // pictures only update at the end (no morphing curves); numbers still tick.
  const frame = () => {
    if (token !== S.nnToken) return;
    if (!S.nnRun) { if (RM) applyNN(S.py.nn_state(S.dirNN)); return; }
    const t0 = performance.now();
    applyNN(S.py.nn_step(Math.min(n, S.nnTarget - S.nn.epochs), S.dirNN), !RM);
    const dt = performance.now() - t0;
    n = Math.max(1, Math.min(80, Math.round(n * NN_FRAME_BUDGET_MS / Math.max(dt, 1))));
    if (S.nn.epochs >= S.nnTarget) { stopNN(); if (RM) applyNN(S.py.nn_state(S.dirNN)); }
    else requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function pickArch(a) {
  stopNN();
  S.arch = a;
  document.querySelectorAll("#arch-pick button").forEach((b) =>
    b.setAttribute("aria-checked", String(b.dataset.arch === a)));
  if (S.py) { applyNN(S.py.nn_reset(a, S.dirNN)); setNNBtn(); }
}

function wireNN() {
  document.querySelectorAll("#arch-pick button").forEach((b) =>
    b.addEventListener("click", () => pickArch(b.dataset.arch)));
  $("nn-train").addEventListener("click", () => (S.nnRun ? stopNN() : startNN()));
  $("nn-reset").addEventListener("click", () => pickArch(S.arch));
}

// =====================================================================
// ④ REPORT CARD
// =====================================================================
const REPORT_ROWS = [
  { key: "line", name: "La linea", sub: "guarda solo la superficie", params: 2 },
  ...Object.entries(ARCH_INFO).map(([key, a]) =>
    ({ key, name: a.name, sub: "guarda superficie, distanza e anno", params: a.params })),
];

function drawReport() {
  const done = REPORT_ROWS.filter((r) => S.report[r.key]);
  const scale = Math.max(140000, ...done.map((r) => S.report[r.key].mae_test));
  const best = done.length > 1
    ? done.reduce((a, b) => (S.report[a.key].mae_test <= S.report[b.key].mae_test ? a : b)).key
    : null;
  $("report").innerHTML =
    `<thead><tr><th>Modello</th><th>Manopole</th><th>Errore sulle case per allenarsi</th>` +
    `<th class="nn-hl">Errore sulle case mai viste</th></tr></thead><tbody>` +
    REPORT_ROWS.map((r) => {
      const v = S.report[r.key];
      const name = `<td><strong>${r.name}</strong>${r.key === best ? ' <span class="best">★ la migliore</span>' : ""}` +
        `<small>${r.sub}</small></td><td class="num">${r.params}</td>`;
      if (!v) return `<tr class="empty">${name}<td colspan="2" class="muted">non ancora allenato</td></tr>`;
      return `<tr>${name}<td class="num">${fmtEur(v.mae_train)}</td>` +
        `<td class="nn-hl"><div class="rcell"><span class="rbar"><i style="width:${(100 * v.mae_test / scale).toFixed(1)}%"></i></span>` +
        `<span class="num">${fmtEur(v.mae_test)}</span></div></td></tr>`;
    }).join("") + `</tbody>`;
}

// =====================================================================
// boot
// =====================================================================
function waitForPython(timeoutMs = 45000) {
  return new Promise((resolve, reject) => {
    if (window.NNPY) return resolve(window.NNPY);
    const t = setTimeout(() => reject(new Error("Python start timeout")), timeoutMs);
    window.addEventListener("nnpy-ready", () => { clearTimeout(t); resolve(window.NNPY); }, { once: true });
  });
}

function setStatus(state, text) {
  $("py-status").dataset.state = state;
  $("py-status-text").textContent = text;
}

function rebuildCharts() {
  drawDataChart();
  buildLineChart();
  buildNNChart();
  drawNet();
}

async function init() {
  mountGlossary();
  S.data = await (await fetch("house_prices.json")).json();
  S.houses = [
    ...S.data.train.map((h) => ({ ...h, test: false })),
    ...S.data.test.map((h) => ({ ...h, test: true })),
  ];
  buildDataTable();
  buildChips($("data-dir"), S.dirData, (j) => { S.dirData = j; drawDataChart(); });
  buildChips($("nn-dir"), S.dirNN, (j) => {
    S.dirNN = j;
    buildNNChart();
    if (S.py) applyNN(S.py.nn_state(j));
  });
  wireLine();
  wireNN();
  document.querySelectorAll("#arch-pick button").forEach((b) =>
    b.setAttribute("aria-checked", String(b.dataset.arch === S.arch)));
  rebuildCharts();
  drawReport();
  $("net-diagram").innerHTML = '<p class="nn-wait">La rete compare quando Python è pronto…</p>';

  let timer;
  window.addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(rebuildCharts, 150);
  });

  fetch("py/nnlab.py").then((r) => r.text())
    .then((t) => { $("py-src").textContent = t; }).catch(() => {});

  try {
    S.py = await waitForPython();
  } catch (e) {
    setStatus("error", "Python non è partito. Serve internet la prima volta (per scaricarlo): " +
      "ricarica la pagina. I dati qui sotto si vedono lo stesso.");
    return;
  }
  setStatus("ready", "Python è pronto: ogni calcolo di questa pagina avviene in Python, dentro il tuo browser.");
  applyLine(S.py.line_state());
  pickArch(S.arch);
  document.querySelectorAll("#sec-line [disabled], #sec-nn [disabled]")
    .forEach((el) => { el.disabled = false; });
}

init().catch(showLoadError);
