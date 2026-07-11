// Word Map Game — "where does it belong?"
// A standalone tab over precomputed layer-0 embedding coordinates
// (web/embedding_game.json). Anchors are drawn as labelled landmarks; the
// learner drags each held-out mystery word onto the map and is scored by how
// close their guess lands to the model's true 2D position. No model, no math
// beyond a distance — all judging is against stored coordinates.
import { showLoadError } from "./aa.js";
import { mountGlossary } from "./glossary.js";

const RM = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const PAD = 30;            // plane inner padding (px)
const FULL_MARKS = 0.045;  // dist < this fraction of the diagonal => 100 pts
const ZERO_MARKS = 0.42;   // dist >= this fraction => 0 pts

const app = {
  data: null, x: null, y: null, W: 0, H: 0, svg: null, gMarks: null,
  rounds: [], i: 0, total: 0, placed: false, diag: 1,
};

// ---------- helpers ----------
const $ = (id) => document.getElementById(id);
const colorOf = (cat) =>
  (app.data.categories.find((c) => c.name === cat) || {}).color || "var(--ink)";
// category names in embedding_game.json stay in English (they're data keys);
// this is only a display-label lookup for what the learner sees on screen.
const CAT_IT = {
  animals: "animali",
  cities: "città",
  emotions: "emozioni",
  colors: "colori",
  food: "cibo",
  family: "famiglia",
};
const catLabel = (name) => CAT_IT[name] || name;

function shuffle(arr) {
  const a = arr.slice();
  for (let k = a.length - 1; k > 0; k--) {
    const j = Math.floor(Math.random() * (k + 1));
    [a[k], a[j]] = [a[j], a[k]];
  }
  return a;
}

// ---------- load + build the plane ----------
async function init() {
  mountGlossary();
  app.data = await (await fetch("embedding_game.json")).json();
  buildPlane();
  buildLegend();
  startGame();

  window.addEventListener("resize", () => {
    // keep coordinates honest if the box resizes: rebuild scales + redraw.
    // Only re-DRAW — the round was scored once, at placement.
    buildPlane();
    if (app.placed && app._guess && app.rounds[app.i]) drawReveal();
  });
}

function buildPlane() {
  const root = $("game-plane");
  app.W = root.clientWidth || 640;
  app.H = root.clientHeight || 480;
  app.diag = Math.hypot(app.W, app.H);
  root.innerHTML = "";

  const pts = [
    ...app.data.anchors.map((d) => d.xy),
    ...app.data.rounds.map((d) => d.xy),
  ];
  app.x = d3.scaleLinear().domain(d3.extent(pts, (d) => d[0]))
    .range([PAD, app.W - PAD]);
  app.y = d3.scaleLinear().domain(d3.extent(pts, (d) => d[1]))
    .range([app.H - PAD, PAD]);

  app.svg = d3.select(root).append("svg")
    .attr("class", "game-svg")
    .attr("viewBox", `0 0 ${app.W} ${app.H}`)
    .attr("width", "100%").attr("height", "100%");

  // category names sit faintly behind their cluster (drawn at the centroid)
  const gLabels = app.svg.append("g").attr("class", "cat-labels");
  for (const c of app.data.categories) {
    const own = app.data.anchors.filter((a) => a.cat === c.name);
    if (!own.length) continue;
    const cx = d3.mean(own, (a) => app.x(a.xy[0]));
    const cy = d3.mean(own, (a) => app.y(a.xy[1]));
    gLabels.append("text").attr("class", "cat-label")
      .attr("x", cx).attr("y", cy).attr("fill", c.color).text(catLabel(c.name));
  }

  app.svg.append("g").attr("class", "anchors");
  app.gMarks = app.svg.append("g").attr("class", "marks");

  // tap-to-place fallback (works on touch + non-draggers)
  app.svg.on("click", (ev) => {
    if (app.placed) return;
    const [lx, ly] = d3.pointer(ev);
    placeAt(lx, ly);
  });

  drawAnchors();
}

function drawAnchors() {
  const sel = app.svg.select(".anchors").selectAll("g.anchor")
    .data(app.data.anchors, (d) => d.word);
  const enter = sel.enter().append("g").attr("class", "anchor");
  enter.append("circle").attr("r", 4);
  enter.append("text").attr("x", 7).attr("y", 3.5);
  // hover raises the dot to the top and reveals its label even if decluttered
  enter.on("pointerenter", function () {
    this.parentNode.appendChild(this);
    this.classList.add("hover");
  }).on("pointerleave", function () {
    this.classList.remove("hover");
  });
  enter.merge(sel)
    .attr("transform", (d) => `translate(${app.x(d.xy[0])},${app.y(d.xy[1])})`)
    .each(function (d) {
      const n = d3.select(this);
      n.select("circle").attr("fill", colorOf(d.cat));
      n.select("text").attr("class", "anchor-label").text(d.word);
    });
  declutterLabels();
}

// Greedy de-overlap: walk labels top-to-bottom and keep one only if its box
// clears every label already kept; the rest are hidden (their dot stays, and
// the word reappears on hover). Turns a pile of overlapping words into a
// readable map.
function declutterLabels() {
  const nodes = app.svg.selectAll(".anchor text").nodes();
  nodes.forEach((t) => t.classList.remove("label-hidden"));
  nodes.sort((a, b) => {
    const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
    return ra.top - rb.top || ra.left - rb.left;
  });
  const placed = [];
  for (const t of nodes) {
    const r = t.getBoundingClientRect();
    const hit = placed.some((p) =>
      !(r.right < p.left - 2 || r.left > p.right + 2 ||
        r.bottom < p.top - 2 || r.top > p.bottom + 2));
    if (hit) t.classList.add("label-hidden");
    else placed.push(r);
  }
}

function buildLegend() {
  const ul = $("cat-legend");
  ul.innerHTML = "";
  for (const c of app.data.categories) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="sw" style="background:${c.color}"></span>${catLabel(c.name)}`;
    ul.appendChild(li);
  }
}

// ---------- game flow ----------
function startGame() {
  app.rounds = shuffle(app.data.rounds);
  app.i = 0;
  app.total = 0;
  app.results = [];
  $("score-total").textContent = "0";
  $("replay-btn").hidden = true;
  loadRound();
}

function loadRound() {
  app.placed = false;
  app._guess = null;
  app.gMarks.selectAll("*").remove();
  app.svg.select(".anchors").classed("dim", false);

  const r = app.rounds[app.i];
  $("round-count").textContent = `${app.i + 1} / ${app.rounds.length}`;
  const chip = $("word-chip");
  chip.textContent = r.word;
  chip.dataset.word = r.word;
  chip.classList.remove("done", "finished");
  chip.style.borderColor = colorOf(r.cat);
  $("tray-hint").textContent = "trascinala sulla mappa — oppure tocca un punto";
  $("tray-hint").hidden = false;
  $("next-btn").disabled = true;
  $("game-feedback").innerHTML =
    `<p class="feedback-empty">Trascina <strong>${r.word}</strong> dove pensi
     che vada, poi rilascia.</p>`;
  renderPips();
}

function renderPips() {
  const wrap = $("progress-pips");
  if (!wrap) return;
  wrap.innerHTML = "";
  app.rounds.forEach((r, idx) => {
    const res = app.results[idx];
    const pip = document.createElement("span");
    pip.className = "pip" + (res ? " done " + res.cls
      : idx === app.i && !app.placed ? " current" : "");
    pip.textContent = res ? res.points : idx + 1;
    pip.title = res ? `${r.word}: +${res.points}` : r.word;
    wrap.appendChild(pip);
  });
}

// the heart of it: place the current word, score it ONCE, reveal the truth
function placeAt(lx, ly) {
  if (app.placed) return;
  app.placed = true;
  const r = app.rounds[app.i];
  const truePx = [app.x(r.xy[0]), app.y(r.xy[1])];
  const dist = Math.hypot(lx - truePx[0], ly - truePx[1]);
  const points = scoreFor(dist);
  // guess kept in DATA coordinates so a resize can redraw it faithfully
  app._guess = { xy: [app.x.invert(lx), app.y.invert(ly)], points, dist };

  app.total += points;
  app.results[app.i] = { points, cls: band(points)[1] };
  $("score-total").textContent = String(app.total);
  $("tray-hint").hidden = true;
  $("word-chip").classList.add("done");
  $("next-btn").disabled = false;
  renderPips();
  renderFeedback(r, points, dist);
  drawReveal();
}

// pure redraw of the placed round's markers (safe to call again on resize)
function drawReveal() {
  const r = app.rounds[app.i];
  const guessPx = [app.x(app._guess.xy[0]), app.y(app._guess.xy[1])];
  const truePx = [app.x(r.xy[0]), app.y(r.xy[1])];

  app.gMarks.selectAll("*").remove();
  app.svg.select(".anchors").classed("dim", true);

  app.gMarks.append("line").attr("class", "guess-line")
    .attr("x1", guessPx[0]).attr("y1", guessPx[1])
    .attr("x2", RM ? truePx[0] : guessPx[0])
    .attr("y2", RM ? truePx[1] : guessPx[1]);
  if (!RM) {
    app.gMarks.select(".guess-line").transition().duration(700)
      .attr("x2", truePx[0]).attr("y2", truePx[1]);
  }

  app.gMarks.append("circle").attr("class", "guess-mark")
    .attr("cx", guessPx[0]).attr("cy", guessPx[1]).attr("r", 9);

  const trueG = app.gMarks.append("g").attr("class", "true-mark")
    .attr("transform", `translate(${truePx[0]},${truePx[1]})`);
  const c = trueG.append("circle").attr("fill", colorOf(r.cat))
    .attr("r", RM ? 7 : 0);
  trueG.append("text").attr("class", "true-label").attr("x", 10).attr("y", 4)
    .attr("fill", colorOf(r.cat)).text(r.word);
  if (!RM) c.transition().delay(650).duration(400).attr("r", 7);
}

function scoreFor(dist) {
  const frac = dist / app.diag;
  if (frac <= FULL_MARKS) return 100;
  const t = Math.max(0, 1 - (frac - FULL_MARKS) / (ZERO_MARKS - FULL_MARKS));
  return Math.round(100 * Math.pow(t, 1.15));
}

function band(points) {
  if (points >= 85) return ["Centro! 🎯", "great"];
  if (points >= 60) return ["Molto vicino!", "good"];
  if (points >= 35) return ["Nella zona giusta.", "ok"];
  if (points >= 15) return ["Un po' fuori — ma si capisce il perché.", "meh"];
  return ["Difficile, eh!", "cold"];
}

function renderFeedback(r, points, dist) {
  const [msg, cls] = band(points);
  const nbr = (r.neighbors || []).slice(0, 3);
  const nbrHtml = nbr.map((n) =>
    `<span class="nbr">${n.word}<span class="aa-sim aa-detail"> ${n.sim.toFixed(2)}</span></span>`
  ).join(", ");
  const note = r.note
    ? `<p class="feedback-note">💡 ${r.note}</p>` : "";
  $("game-feedback").innerHTML = `
    <p class="feedback-head ${cls}">${msg} <span class="pts">+${points}</span></p>
    <p class="feedback-sub">Il modello ha messo <strong>${r.word}</strong>
      più vicino a ${nbrHtml || "il resto della mappa"}.</p>
    <p class="aa-detail feedback-dist">Il tuo tentativo era a ${Math.round(dist)} px
      dalla posizione vera.</p>
    ${note}`;
}

function nextRound() {
  app.i++;
  if (app.i >= app.rounds.length) return endGame();
  loadRound();
}

function endGame() {
  const max = app.rounds.length * 100;
  app.svg.select(".anchors").classed("dim", false);
  const chip = $("word-chip");
  chip.classList.remove("done");
  chip.classList.add("finished");
  chip.textContent = "🎉 finito!";
  $("tray-hint").hidden = true;
  $("round-count").textContent = `${app.rounds.length} / ${app.rounds.length}`;
  $("next-btn").disabled = true;
  $("replay-btn").hidden = false;
  const pct = Math.round((app.total / max) * 100);
  const verdict = pct >= 80 ? "Leggi la mappa come un professionista."
    : pct >= 55 ? "Hai un buon intuito per il significato delle parole."
    : "Ora hai visto come il modello raggruppa i significati.";
  $("game-feedback").innerHTML = `
    <p class="feedback-head great">Punteggio finale: ${app.total} / ${max}</p>
    <p class="feedback-sub">${verdict}</p>
    <p class="feedback-sub">I significati simili stanno vicini tra loro — è
      proprio questa l'idea alla base di un embedding.</p>`;
  renderPips();
}

// ---------- drag (pointer-capture, with a floating ghost chip) ----------
// setPointerCapture routes every move/up for this pointer back to the chip even
// while the cursor is over the map, so the drop always registers.
function initDrag() {
  const chip = $("word-chip");
  let ghost = null, dragging = false;

  const moveGhost = (ev) => {
    if (!ghost) return;
    ghost.style.left = ev.clientX + "px";
    ghost.style.top = ev.clientY + "px";
  };
  const end = (ev) => {
    if (!dragging) return;
    dragging = false;
    chip.classList.remove("dragging");
    if (ghost) { ghost.remove(); ghost = null; }
    const rect = $("game-plane").getBoundingClientRect();
    const lx = ev.clientX - rect.left, ly = ev.clientY - rect.top;
    if (lx >= 0 && ly >= 0 && lx <= rect.width && ly <= rect.height) {
      placeAt(lx, ly);
    }
  };

  chip.addEventListener("pointerdown", (ev) => {
    if (app.placed) return;
    ev.preventDefault();
    dragging = true;
    try { chip.setPointerCapture(ev.pointerId); } catch (e) { /* older browsers */ }
    chip.classList.add("dragging");
    ghost = document.createElement("span");
    ghost.className = "word-chip ghost-chip";
    ghost.textContent = chip.dataset.word || chip.textContent;
    ghost.style.borderColor = chip.style.borderColor;
    moveGhost(ev);
    document.body.appendChild(ghost);
  });
  chip.addEventListener("pointermove", (ev) => { if (dragging) moveGhost(ev); });
  chip.addEventListener("pointerup", end);
  chip.addEventListener("pointercancel", end);
}

$("next-btn").addEventListener("click", nextRound);
$("replay-btn").addEventListener("click", startGame);
initDrag();
init().catch(showLoadError);
