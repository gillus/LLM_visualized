// \u00abLa mappa dei significati\u00bb -- attention explainer scene, pre-transpiled from JSX
// offline. Sentence: 'Ai rigori finisce la partita di calcio'; shows attention moving
// the ambiguous 'calcio' vector toward its sport sense. Registers window.SceneMappa.
/* Versione A — «La mappa dei significati»
   La frase entra; «calcio» è un punto ambiguo nello spazio dei significati;
   l'attenzione (dentro il blocco) lo sposta verso il senso giusto.
   Capitoli: 0 Ambiguità · 10 La mappa · 24 Attenzione · 40 Lo spostamento · 53 Gli strati */
/* global React, ReactDOM, Stage, useTime, useTimeline, Easing, interpolate, animate, clamp */

const C = {
  bg: '#F0EEE6',
  card: '#FAF9F5',
  ink: '#1F1E1A',
  inkSoft: 'rgba(31,30,26,0.58)',
  inkFaint: 'rgba(31,30,26,0.30)',
  line: 'rgba(31,30,26,0.16)',
  coral: '#C15F3C',
  coralBright: '#D97757',
  coralSoft: 'rgba(193,95,60,0.14)',
  sport: '#4E7E5A',
  chem: '#5B7EA6',
  hit: '#A9853C'
};
const SERIF = "'Fraunces', Georgia, serif";
const SANS = "'Atkinson Hyperlegible', system-ui, sans-serif";
const MONO = "'Spline Sans Mono', ui-monospace, monospace";
const TOKENS = ['Ai', 'rigori', 'finisce', 'la', 'partita', 'di', 'calcio'];
const CI = 6; // indice di «calcio» (ultima parola: attenzione solo all'indietro)
const WEIGHTS = [0.04, 0.33, 0.12, 0.05, 0.40, 0.06, null];
const ORDER = [4, 1, 2, 5, 3, 0]; // per peso decrescente
const BARS = [[0.35, 0.5, 0.3, 0.45, 0.55, 0.35, 0.45], [0.75, 0.25, 0.7, 0.3, 0.6, 0.45, 0.8], [0.6, 0.35, 0.5, 0.7, 0.3, 0.55, 0.4], [0.25, 0.5, 0.3, 0.6, 0.35, 0.5, 0.3], [0.7, 0.3, 0.8, 0.2, 0.65, 0.4, 0.75], [0.3, 0.45, 0.25, 0.55, 0.4, 0.3, 0.5], [0.5, 0.48, 0.52, 0.47, 0.5, 0.49, 0.51] // calcio: piatto = indeciso
];
const CHAPTERS = [{
  t0: 0,
  title: 'Ambiguità'
}, {
  t0: 10,
  title: 'La mappa'
}, {
  t0: 24,
  title: 'Attenzione'
}, {
  t0: 40,
  title: 'Lo spostamento'
}, {
  t0: 53,
  title: 'Gli strati'
}];
const DUR = 64;
const boolProp = (v, d) => v == null || v === '' ? d : !(v === false || v === 'false');
function fadeInOut(t, t0, t1, fin = 0.5, fout = 0.5) {
  if (t < t0 || t > t1) return 0;
  if (t < t0 + fin) return Easing.easeOutCubic((t - t0) / fin);
  if (t > t1 - fout) return 1 - Easing.easeInCubic((t - (t1 - fout)) / fout);
  return 1;
}
function qPoint(p, x0, y0, cx, cy, x1, y1) {
  const u = 1 - p;
  return {
    x: u * u * x0 + 2 * u * p * cx + p * p * x1,
    y: u * u * y0 + 2 * u * p * cy + p * p * y1
  };
}
function rowLayout(size) {
  const gap = size * 0.42;
  const widths = TOKENS.map(w => w.length * size * 0.56 + size * 1.35);
  const total = widths.reduce((a, b) => a + b, 0) + gap * (TOKENS.length - 1);
  let x = (1600 - total) / 2;
  return TOKENS.map((w, i) => {
    const o = {
      w: widths[i],
      x,
      cx: x + widths[i] / 2
    };
    x += widths[i] + gap;
    return o;
  });
}
const EASE_ROW = [Easing.linear, Easing.easeInOutCubic, Easing.linear, Easing.easeInOutCubic, Easing.linear, Easing.easeInOutCubic];
const ROW_T = [0, 8.6, 10, 24, 26, 38.2, 39.8];
const rowYF = interpolate(ROW_T, [430, 430, 92, 92, 470, 470, 92], EASE_ROW);
const rowSizeF = interpolate(ROW_T, [40, 40, 24, 24, 38, 38, 24], EASE_ROW);
const glowF = t => (fadeInOut(t, 4.6, 8.4, 0.6, 0.6) + fadeInOut(t, 26.2, 37.6, 0.8, 0.8)) * (0.7 + 0.3 * Math.sin(t * 3));

// ── piccoli pezzi condivisi ─────────────────────────────────────────────────

function W({
  c,
  children
}) {
  return /*#__PURE__*/React.createElement("span", {
    style: {
      color: c || C.coral,
      fontWeight: 600
    }
  }, children);
}
function Caption({
  t0,
  t1,
  children
}) {
  const t = useTime();
  const o = fadeInOut(t, t0, t1, 0.5, 0.45);
  if (o <= 0) return null;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 100,
      right: 100,
      top: 806,
      textAlign: 'center',
      opacity: o,
      transform: `translateY(${(1 - o) * 10}px)`,
      fontFamily: SANS,
      fontSize: 28,
      fontWeight: 500,
      color: C.inkSoft
    }
  }, children);
}
function ChapterPause({
  markers
}) {
  const {
    time,
    playing,
    setTime,
    setPlaying
  } = useTimeline();
  const prev = React.useRef(time);
  React.useEffect(() => {
    const p = prev.current;
    prev.current = time;
    if (!playing) return;
    for (const m of markers) {
      if (p < m && time >= m) {
        setPlaying(false);
        setTime(m);
        break;
      }
    }
  }, [time, playing]);
  return null;
}
function ChapterBar() {
  const {
    time,
    setTime,
    setPlaying
  } = useTimeline();
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      bottom: 14,
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      gap: 8,
      zIndex: 40
    }
  }, CHAPTERS.map((c, i) => {
    const next = CHAPTERS[i + 1] ? CHAPTERS[i + 1].t0 : 1e9;
    const active = time >= c.t0 && time < next;
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      onClick: () => {
        setPlaying(false);
        setTime(c.t0 === 0 ? 0.03 : c.t0 + 0.001);
      },
      style: {
        cursor: 'pointer',
        padding: '7px 16px',
        borderRadius: 999,
        fontFamily: SANS,
        fontSize: 16,
        fontWeight: 500,
        background: active ? C.ink : 'transparent',
        color: active ? C.bg : C.inkSoft,
        border: `1px solid ${active ? C.ink : C.line}`
      }
    }, i + 1 + ' · ' + c.title);
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: MONO,
      fontSize: 13,
      color: C.inkFaint,
      marginLeft: 10,
      whiteSpace: 'nowrap'
    }
  }, "spazio \u25B7 avanti"));
}
function StartOverlay() {
  const {
    time,
    playing,
    setPlaying
  } = useTimeline();
  if (time > 0.02 || playing) return null;
  return /*#__PURE__*/React.createElement("div", {
    onClick: () => setPlaying(true),
    style: {
      position: 'absolute',
      inset: 0,
      background: C.bg,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 26,
      zIndex: 50,
      cursor: 'pointer'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: MONO,
      fontSize: 16,
      letterSpacing: '0.2em',
      color: C.inkSoft
    }
  }, "COME FUNZIONA UN TRANSFORMER"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: SERIF,
      fontSize: 58,
      fontWeight: 600,
      color: C.ink,
      textAlign: 'center',
      maxWidth: 1000,
      lineHeight: 1.15
    }
  }, "L'attenzione: come una parola trova il suo significato"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: SERIF,
      fontSize: 28,
      fontStyle: 'italic',
      color: C.inkSoft
    }
  }, "\xABAi rigori finisce la partita di calcio\xBB"), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 14,
      fontFamily: SANS,
      fontSize: 20,
      color: C.bg,
      background: C.ink,
      padding: '12px 26px',
      borderRadius: 999
    }
  }, "premi SPAZIO o tocca qui \u2014 si ferma da solo a fine capitolo"));
}
function ScreenLabel() {
  const t = Math.floor(useTime());
  React.useEffect(() => {
    const el = document.querySelector('svg[data-om-exportable-video-with-duration-secs]');
    if (el) el.setAttribute('data-screen-label', 't=' + t + 's');
  }, [t]);
  return null;
}
function VBars({
  heights,
  color,
  o
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 4,
      justifyContent: 'center',
      marginTop: 10,
      opacity: o,
      height: 32,
      alignItems: 'flex-end'
    }
  }, heights.map((h, j) => /*#__PURE__*/React.createElement("div", {
    key: j,
    style: {
      width: 7,
      height: 4 + h * 26,
      background: color,
      borderRadius: 2
    }
  })));
}

// ── capitolo 1: titolo + sensi ──────────────────────────────────────────────

function TitleBlock({
  t
}) {
  const o = fadeInOut(t, 0.35, 8.4, 0.9, 0.9);
  if (o <= 0) return null;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 0,
      opacity: o,
      transform: `translateY(${(1 - o) * 12}px)`
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 158,
      textAlign: 'center',
      fontFamily: MONO,
      fontSize: 15,
      letterSpacing: '0.2em',
      color: C.inkFaint
    }
  }, "CAPITOLO 1"), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 192,
      textAlign: 'center',
      fontFamily: SERIF,
      fontSize: 60,
      fontWeight: 600,
      color: C.ink
    }
  }, "Una parola, tanti significati."));
}
function SenseTags({
  t
}) {
  const defs = [{
    label: 'elemento chimico',
    color: C.chem,
    x: 930,
    y: 318,
    t0: 5.0
  }, {
    label: 'lo sport',
    color: C.sport,
    x: 1330,
    y: 322,
    t0: 5.45
  }, {
    label: 'un colpo',
    color: C.hit,
    x: 1150,
    y: 610,
    t0: 5.9
  }];
  const cx = 1304,
    cy = 462;
  const anyO = fadeInOut(t, 5.0, 8.9, 0.5, 0.6);
  if (anyO <= 0) return null;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      pointerEvents: 'none'
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: 1600,
    height: 900,
    style: {
      position: 'absolute',
      left: 0,
      top: 0
    }
  }, defs.map((d, i) => /*#__PURE__*/React.createElement("line", {
    key: i,
    x1: cx,
    y1: cy,
    x2: d.x,
    y2: d.y,
    stroke: d.color,
    strokeWidth: 1.5,
    strokeDasharray: "3 7",
    opacity: fadeInOut(t, d.t0, 8.9, 0.5, 0.6) * 0.55
  }))), defs.map((d, i) => {
    const o = fadeInOut(t, d.t0, 8.9, 0.5, 0.6);
    if (o <= 0) return null;
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        position: 'absolute',
        left: d.x,
        top: d.y,
        transform: `translate(-50%,-50%) scale(${0.9 + 0.1 * o})`,
        opacity: o,
        display: 'flex',
        alignItems: 'center',
        gap: 9,
        background: C.card,
        border: `1.5px solid ${d.color}`,
        borderRadius: 999,
        padding: '8px 18px'
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        width: 10,
        height: 10,
        borderRadius: 5,
        background: d.color
      }
    }), /*#__PURE__*/React.createElement("div", {
      style: {
        fontFamily: SANS,
        fontSize: 21,
        color: C.ink
      }
    }, d.label));
  }));
}

// ── la frase (riga di token, persiste per tutta l'animazione) ──────────────

function TokenRow({
  t
}) {
  const size = rowSizeF(t),
    y = rowYF(t);
  const chips = rowLayout(size);
  const recolor = animate({
    from: 0,
    to: 1,
    start: 37.8,
    end: 39.2
  })(t);
  const barsO = fadeInOut(t, 3.6, 8.3, 0.6, 0.6);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      pointerEvents: 'none'
    }
  }, TOKENS.map((w, i) => {
    const o = fadeInOut(t, 2.0 + i * 0.28, 1e9, 0.55, 1);
    if (o <= 0) return null;
    const isC = i === CI;
    const glow = isC ? glowF(t) : 0;
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        position: 'absolute',
        left: chips[i].cx,
        top: y + (1 - o) * 14,
        transform: 'translateX(-50%)',
        opacity: o
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        width: chips[i].w,
        height: size * 1.55,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: isC ? `rgba(193,95,60,${0.05 + 0.16 * recolor})` : C.card,
        border: `${isC ? 2 : 1.5}px solid ${isC ? C.coral : C.line}`,
        borderRadius: size * 0.3,
        boxShadow: glow > 0.05 ? `0 0 ${18 * glow}px rgba(193,95,60,${0.4 * glow})` : '0 1px 3px rgba(31,30,26,0.06)',
        fontFamily: SERIF,
        fontSize: size,
        fontWeight: 500,
        color: isC ? C.coral : C.ink
      }
    }, w), barsO > 0 && size > 32 && /*#__PURE__*/React.createElement(VBars, {
      heights: BARS[i],
      color: isC ? 'rgba(31,30,26,0.45)' : C.inkFaint,
      o: barsO
    }));
  }));
}

// ── capitolo 3: blocco + fasci di attenzione ────────────────────────────────

function BlockStack({
  t
}) {
  const o = fadeInOut(t, 24.5, 39.5, 1.1, 1.0);
  if (o <= 0) return null;
  const boxes = [{
    dx: 36,
    dy: -36,
    op: 0.22
  }, {
    dx: 18,
    dy: -18,
    op: 0.5
  }, {
    dx: 0,
    dy: 0,
    op: 1
  }];
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: o,
      pointerEvents: 'none'
    }
  }, boxes.map((b, k) => /*#__PURE__*/React.createElement("div", {
    key: k,
    style: {
      position: 'absolute',
      left: 282 + b.dx,
      top: 302 + b.dy,
      width: 1036,
      height: 266,
      background: C.card,
      border: `1.5px solid ${C.line}`,
      borderRadius: 20,
      opacity: b.op,
      boxShadow: b.dx === 0 ? '0 10px 34px rgba(31,30,26,0.08)' : 'none'
    }
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 288,
      top: 258,
      fontFamily: MONO,
      fontSize: 15,
      letterSpacing: '0.16em',
      color: C.inkSoft
    }
  }, "BLOCCO TRANSFORMER"), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 282,
      top: 258,
      width: 1030,
      textAlign: 'right',
      fontFamily: MONO,
      fontSize: 14,
      color: C.inkFaint,
      whiteSpace: 'nowrap'
    }
  }, "\u2190 attenzione solo all'indietro"));
}
function Beams({
  t,
  showWeights
}) {
  const o = fadeInOut(t, 25.9, 38.8, 0.6, 0.9);
  if (o <= 0) return null;
  const chips = rowLayout(38);
  const yA = 464;
  const cxC = chips[CI].cx;
  const beams = ORDER.map((i, r) => {
    const w = WEIGHTS[i];
    const start = 26.4 + r * 0.5;
    return {
      i,
      r,
      w,
      start,
      drawP: animate({
        from: 0,
        to: 1,
        start,
        end: start + 1.0,
        ease: Easing.easeInOutCubic
      })(t),
      x1: chips[i].cx,
      ctrlX: (cxC + chips[i].cx) / 2,
      ctrlY: yA - (60 + w * 420)
    };
  });
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: o,
      pointerEvents: 'none'
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: 1600,
    height: 900,
    style: {
      position: 'absolute',
      left: 0,
      top: 0
    }
  }, beams.map(b => b.drawP > 0 && /*#__PURE__*/React.createElement("path", {
    key: b.i,
    d: `M ${cxC} ${yA} Q ${b.ctrlX} ${b.ctrlY} ${b.x1} ${yA}`,
    fill: "none",
    stroke: C.coral,
    strokeWidth: 1.5 + b.w * 24,
    strokeLinecap: "round",
    opacity: Math.min(0.9, 0.3 + b.w * 1.5),
    pathLength: 1,
    strokeDasharray: 1,
    strokeDashoffset: 1 - b.drawP
  })), t > 30 && beams.slice(0, 3).map(b => {
    const p = (t * 0.45 + b.r * 0.37) % 1;
    const pos = qPoint(p, b.x1, yA, b.ctrlX, b.ctrlY, cxC, yA);
    const po = fadeInOut(t, 30.2, 36.8, 0.8, 0.8) * Math.min(1, p * 4, (1 - p) * 4 + 0.2);
    return /*#__PURE__*/React.createElement("circle", {
      key: 'd' + b.i,
      cx: pos.x,
      cy: pos.y,
      r: 4 + b.w * 8,
      fill: C.coralBright,
      opacity: po
    });
  })), showWeights && beams.map(b => {
    const lo = fadeInOut(t, b.start + 0.85, 37.9, 0.4, 0.5);
    if (lo <= 0) return null;
    return /*#__PURE__*/React.createElement("div", {
      key: 'l' + b.i,
      style: {
        position: 'absolute',
        left: b.ctrlX,
        top: (yA + b.ctrlY) / 2 - 30,
        transform: 'translateX(-50%)',
        opacity: lo,
        fontFamily: MONO,
        fontSize: 18,
        fontWeight: 500,
        color: C.coral,
        background: 'rgba(250,249,245,0.85)',
        padding: '1px 7px',
        borderRadius: 8
      }
    }, Math.round(b.w * 100) + '%');
  }));
}

// ── la mappa dei significati (cap. 2, 4, 5) ────────────────────────────────

const CLUSTERS = [{
  name: 'CHIMICA',
  color: C.chem,
  x: 470,
  y: 320,
  words: [['latte', -85, -50], ['osso', 70, -62], ['minerale', -30, 58], ['ferro', 95, 26]]
}, {
  name: 'UN COLPO',
  color: C.hit,
  x: 495,
  y: 648,
  words: [['pugno', -70, -40], ['spinta', 80, -18], ['schiaffo', -6, 52]]
}, {
  name: 'SPORT',
  color: C.sport,
  x: 1125,
  y: 452,
  words: [['partita', -100, -62], ['gol', 82, -70], ['stadio', -52, 72], ['pallone', 108, 34], ['rigori', 30, -6]]
}];
const P0 = {
  x: 772,
  y: 470
};
const P1 = {
  x: 1008,
  y: 466
};
function pointPos(t) {
  if (t <= 41.5) {
    const a = fadeInOut(t, 14.6, 41.5, 0.8, 0.7);
    return {
      x: P0.x + Math.sin(t * 1.7) * 9 * a,
      y: P0.y + Math.cos(t * 2.3) * 7 * a
    };
  }
  if (t < 44.5) {
    const p = Easing.easeInOutCubic((t - 41.5) / 3);
    return qPoint(p, P0.x, P0.y, 886, 384, P1.x, P1.y);
  }
  const x = interpolate([44.5, 53.8, 54.9, 55.7, 56.6], [P1.x, P1.x, 1056, 1056, 1080], Easing.easeInOutCubic)(t);
  const y = interpolate([44.5, 53.8, 54.9, 55.7, 56.6], [P1.y, P1.y, 474, 474, 454], Easing.easeInOutCubic)(t);
  return {
    x,
    y
  };
}
function MapCard({
  t
}) {
  const enter = fadeInOut(t, 10.4, 1e9, 1.2, 1);
  if (enter <= 0) return null;
  const dim = interpolate([24, 25.4, 38.6, 39.8], [1, 0.1, 0.1, 1])(t);
  const zoom = interpolate([41, 44.6, 53.2, 55.2], [1, 1.15, 1.15, 1], Easing.easeInOutCubic)(t);
  const pt = pointPos(t);
  const ringA = fadeInOut(t, 14.8, 41.3, 0.8, 0.7);
  const ptO = fadeInOut(t, 14.4, 1e9, 0.5, 1);
  const moveP = clamp((t - 41.5) / 3, 0, 1);
  const gridV = [];
  for (let gx = 300; gx <= 1300; gx += 100) gridV.push(gx);
  const gridH = [];
  for (let gy = 230; gy <= 750; gy += 104) gridH.push(gy);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: enter * dim,
      pointerEvents: 'none'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      transform: `scale(${zoom}) translateY(${(1 - enter) * 26}px)`,
      transformOrigin: '1008px 466px'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 200,
      top: 150,
      width: 1200,
      height: 648,
      background: C.card,
      border: `1.5px solid ${C.line}`,
      borderRadius: 24,
      boxShadow: '0 8px 30px rgba(31,30,26,0.06)'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 234,
      top: 176,
      fontFamily: MONO,
      fontSize: 15,
      letterSpacing: '0.16em',
      color: C.inkFaint
    }
  }, "LO SPAZIO DEI SIGNIFICATI"), /*#__PURE__*/React.createElement("svg", {
    width: 1600,
    height: 900,
    style: {
      position: 'absolute',
      left: 0,
      top: 0
    }
  }, gridV.map(gx => /*#__PURE__*/React.createElement("line", {
    key: 'v' + gx,
    x1: gx,
    y1: 166,
    x2: gx,
    y2: 782,
    stroke: C.ink,
    opacity: 0.04
  })), gridH.map(gy => /*#__PURE__*/React.createElement("line", {
    key: 'h' + gy,
    x1: 216,
    y1: gy,
    x2: 1384,
    y2: gy,
    stroke: C.ink,
    opacity: 0.04
  })), CLUSTERS.map((cl, i) => /*#__PURE__*/React.createElement("line", {
    key: 'c' + i,
    x1: pt.x,
    y1: pt.y,
    x2: cl.x,
    y2: cl.y,
    stroke: cl.color,
    strokeWidth: 1.5,
    strokeDasharray: "4 8",
    opacity: 0.4 * fadeInOut(t, 15.7, 24.0, 0.7, 0.8) * (0.75 + 0.25 * Math.sin(t * 2.2 + i * 2.1))
  })), t > 41.55 && /*#__PURE__*/React.createElement("path", {
    d: `M ${P0.x} ${P0.y} Q 886 384 ${P1.x} ${P1.y}`,
    fill: "none",
    stroke: C.coral,
    strokeWidth: 2.5,
    opacity: 0.55 * (1 - clamp((t - 53.8) / 1.1, 0, 1)),
    pathLength: 1,
    strokeDasharray: 1,
    strokeDashoffset: 1 - Easing.easeInOutCubic(moveP)
  })), CLUSTERS.map((cl, i) => {
    const co = fadeInOut(t, 11.4 + i * 0.5, 1e9, 0.8, 1);
    if (co <= 0) return null;
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        position: 'absolute',
        left: 0,
        top: 0,
        opacity: co
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        position: 'absolute',
        left: cl.x - 160,
        top: cl.y - 120,
        width: 320,
        height: 240,
        borderRadius: '50%',
        background: `radial-gradient(closest-side, ${cl.color}26, transparent)`
      }
    }), /*#__PURE__*/React.createElement("div", {
      style: {
        position: 'absolute',
        left: cl.x,
        top: cl.y - 108,
        transform: 'translateX(-50%)',
        fontFamily: MONO,
        fontSize: 14,
        letterSpacing: '0.18em',
        color: cl.color
      }
    }, cl.name), cl.words.map(([w, dx, dy], j) => /*#__PURE__*/React.createElement("div", {
      key: j,
      style: {
        position: 'absolute',
        left: cl.x + dx,
        top: cl.y + dy
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        width: 9,
        height: 9,
        borderRadius: 5,
        background: cl.color,
        opacity: 0.55
      }
    }), /*#__PURE__*/React.createElement("div", {
      style: {
        position: 'absolute',
        left: 13,
        top: -5,
        fontFamily: MONO,
        fontSize: 16,
        color: 'rgba(31,30,26,0.62)',
        whiteSpace: 'nowrap'
      }
    }, w))));
  }), t > 41.7 && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: P0.x,
      top: P0.y,
      opacity: fadeInOut(t, 41.7, 1e9, 0.6, 1)
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: -9,
      top: -9,
      width: 18,
      height: 18,
      borderRadius: 9,
      border: `2px dashed ${C.inkFaint}`
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: -20,
      top: 16,
      fontFamily: SANS,
      fontSize: 17,
      color: C.inkSoft
    }
  }, "prima")), t > 44.8 && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: P1.x - 165,
      top: P1.y + 40,
      opacity: fadeInOut(t, 44.8, 1e9, 0.6, 1),
      fontFamily: SANS,
      fontSize: 17,
      color: C.coral,
      fontWeight: 600,
      whiteSpace: 'nowrap'
    }
  }, "dopo il blocco"), ptO > 0 && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: pt.x,
      top: pt.y,
      opacity: ptO
    }
  }, ringA > 0 && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: -(22 + Math.sin(t * 2) * 4),
      top: -(22 + Math.sin(t * 2) * 4),
      width: (22 + Math.sin(t * 2) * 4) * 2,
      height: (22 + Math.sin(t * 2) * 4) * 2,
      borderRadius: '50%',
      border: `1.5px solid ${C.coral}`,
      opacity: 0.5 * ringA
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: -11,
      top: -11,
      width: 22,
      height: 22,
      borderRadius: 11,
      background: C.coral,
      border: '3px solid #FFFDF9',
      boxShadow: '0 2px 8px rgba(193,95,60,0.4)'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: -32,
      top: 18,
      fontFamily: SERIF,
      fontSize: 23,
      fontWeight: 600,
      color: C.coral
    }
  }, "calcio")), t > 53.6 && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 880,
      top: 692,
      opacity: fadeInOut(t, 53.6, 1e9, 0.8, 1)
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 8
    }
  }, [0, 1, 2].map(k => /*#__PURE__*/React.createElement(React.Fragment, {
    key: k
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      width: 46,
      height: 34,
      border: `1.5px solid ${C.inkSoft}`,
      borderRadius: 8,
      background: C.bg
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: SANS,
      fontSize: 18,
      color: C.inkSoft
    }
  }, "\u2192"))), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: MONO,
      fontSize: 18,
      color: C.ink,
      whiteSpace: 'nowrap'
    }
  }, "\xD7 12 strati")), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 8,
      fontFamily: SANS,
      fontSize: 16,
      color: C.inkSoft,
      whiteSpace: 'nowrap'
    }
  }, "lo stesso passaggio, ripetuto: il punto si assesta"))));
}

// ── scena ───────────────────────────────────────────────────────────────────

function SceneContent({
  pauseAtChapters,
  showWeights,
  showChapterBar
}) {
  const t = useTime();
  return /*#__PURE__*/React.createElement(React.Fragment, null, pauseAtChapters && /*#__PURE__*/React.createElement(ChapterPause, {
    markers: [10, 24, 40, 53]
  }), /*#__PURE__*/React.createElement(ScreenLabel, null), /*#__PURE__*/React.createElement(TitleBlock, {
    t: t
  }), /*#__PURE__*/React.createElement(MapCard, {
    t: t
  }), /*#__PURE__*/React.createElement(BlockStack, {
    t: t
  }), /*#__PURE__*/React.createElement(Beams, {
    t: t,
    showWeights: showWeights
  }), /*#__PURE__*/React.createElement(TokenRow, {
    t: t
  }), /*#__PURE__*/React.createElement(SenseTags, {
    t: t
  }), /*#__PURE__*/React.createElement(Caption, {
    t0: 6.2,
    t1: 9.4
  }, "Prima del blocco, ", /*#__PURE__*/React.createElement(W, null, "\xABcalcio\xBB"), " \xE8 un vettore ambiguo."), /*#__PURE__*/React.createElement(Caption, {
    t0: 16.0,
    t1: 23.6
  }, "Ogni parola \xE8 un punto nello spazio. ", /*#__PURE__*/React.createElement(W, null, "\xABcalcio\xBB"), " resta a met\xE0 strada."), /*#__PURE__*/React.createElement(Caption, {
    t0: 27.0,
    t1: 32.2
  }, /*#__PURE__*/React.createElement(W, null, "\xABcalcio\xBB"), " guarda indietro, alle parole gi\xE0 lette\u2026"), /*#__PURE__*/React.createElement(Caption, {
    t0: 32.6,
    t1: 37.8
  }, "\u2026e d\xE0 pi\xF9 peso a ", /*#__PURE__*/React.createElement(W, {
    c: C.sport
  }, "\xABpartita\xBB"), " e ", /*#__PURE__*/React.createElement(W, {
    c: C.sport
  }, "\xABrigori\xBB"), "."), /*#__PURE__*/React.createElement(Caption, {
    t0: 45.2,
    t1: 52.6
  }, "Il vettore esce spostato: ora vuol dire ", /*#__PURE__*/React.createElement(W, {
    c: C.sport
  }, "calcio-sport"), "."), /*#__PURE__*/React.createElement(Caption, {
    t0: 57.4,
    t1: 63.6
  }, "Strato dopo strato, ogni parola assorbe il contesto."), showChapterBar && /*#__PURE__*/React.createElement(ChapterBar, null), /*#__PURE__*/React.createElement(StartOverlay, null));
}
function SceneMappa(props) {
  return /*#__PURE__*/React.createElement(Stage, {
    width: 1600,
    height: 900,
    duration: DUR,
    background: C.bg,
    autoplay: false,
    loop: false,
    persistKey: "attn-mappa"
  }, /*#__PURE__*/React.createElement(SceneContent, {
    pauseAtChapters: boolProp(props.pauseAtChapters, true),
    showWeights: boolProp(props.showWeights, true),
    showChapterBar: boolProp(props.showChapterBar, true)
  }));
}
window.SceneMappa = SceneMappa;