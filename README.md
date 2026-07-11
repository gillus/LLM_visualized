# 🧠 How an AI Predicts the Next Word

An interactive scrollytelling visualization of next-token prediction in a tiny
language model (SmolLM2-135M), built for **middle- and high-school teachers and
students**. You scroll through the real pipeline:

**Tokens → Embedding (meaning map) → Layers (logit lens) → Attention → Prediction**

The model runs **offline once** per example; the website is pure static files that
render precomputed JSON — no backend, near-zero RAM, free to host.

## Quick start (no model needed)

```bash
python3 generate/fixture.py            # writes a synthetic demo example
cd web && python3 -m http.server 8000  # → http://localhost:8000
```

## Real data from SmolLM2

```bash
pip install -r requirements.txt
python generate/generate.py            # generates every example in generate/examples.py
```

## Add an example

Edit `generate/examples.py`, then `python generate/generate.py <id>`. Keep prompts
short (≤ ~12 tokens). See `CLAUDE.md` for architecture and `web/SCHEMA.md` for the
data format.

## Deploy

`web/` is fully static. Push it to GitHub Pages / Netlify / Cloudflare Pages.

## Stack

Vanilla ES modules + [D3](https://d3js.org) + [Scrollama](https://github.com/russellsamora/scrollama)
(both via CDN). No build step.
