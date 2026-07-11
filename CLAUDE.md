# LLM_visualized

An interactive **scrollytelling** visualization of how a language model predicts
the next word, built around a tiny model (**SmolLM2-135M**). Audience: middle- and
high-school **teachers and students** learning AI. The story walks the real
inference pipeline: **Tokens → Embedding → Layers → Prediction** (attention has
its own standalone tab, `attention.html`, plus the Architecture page's lanes).

## Architecture: precompute offline, render static

The single most important design decision: **the model never runs in the browser.**

1. `generate/generate.py` runs SmolLM2 once per curated example and dumps a
   precomputed JSON file (`web/examples/<id>.json`) containing everything the UI
   needs — tokens, 2D embedding coords, per-layer logit-lens guesses, attention
   matrices, final probabilities.
2. `web/` is a **pure static site** that fetches that JSON and renders it. No
   backend, no inference, near-zero RAM. Deploy anywhere static (GitHub Pages,
   Netlify, Cloudflare Pages) for $0.

This is why the data **schema is the contract** between the two halves. It's
documented in `web/SCHEMA.md` — read it before touching either side.

## Directory layout

```
generate/
  generate.py        # offline SmolLM2 dump → web/examples/*.json + manifest.json
  examples.py        # the curated list of prompts (edit this to add examples)
  fixture.py         # writes a SYNTHETIC cat-sat.json so the site runs w/o the model
  train_tuned_lens.py # trains a tuned lens → tuned_lens.pt (optional, denoises mid layers)
  tuned_lens_model.py # shared TunedLens definition (trainer + generator import it)
  projection.pkl     # cached PCA transform (created on first run; shared by all examples)
  tuned_lens.pt      # cached tuned lens (created by train_tuned_lens.py; optional)
web/
  index.html      # scrolly sections + sticky chrome (CDN: d3, scrollama)
  attention.html  # "Come funziona l'attenzione?" — 60s guided animation (CDN: React UMD)
  css/style.css
  js/
    main.js       # orchestrator: load example, mount stages, wire Scrollama, state
    decode.js     # pure helpers: decode ragged_q8 attention, softmax(T), interpolation
    stages/       # one renderer per pipeline stage (tokens, embedding, layers,
                  #   prediction; attention.js is kept on disk but no longer
                  #   imported — the story dropped its attention section 2026-07)
    anim/         # attention.html's timeline engine + scene, PRE-TRANSPILED plain JS
                  #   (authored as JSX in a claude.ai artifact; edit the .js directly
                  #   or re-transpile offline — no Babel in the browser)
  examples/
    manifest.json # [{id,title,prompt}] → header dropdown
    <id>.json     # one precomputed example per file
  SCHEMA.md       # the JSON contract (READ THIS)
```

## Dev commands

```bash
# 1. Run the site with the synthetic fixture (no model download needed):
python3 generate/fixture.py          # writes web/examples/cat-sat.json
python3 generate/embedding_game.py --fixture  # writes web/embedding_game.json (Word Map Game tab)
python3 generate/training.py --fixture        # writes web/training.json ("How it's trained" tab)
cd web && python3 -m http.server 8000 # open http://localhost:8000

# 2. Generate REAL data from SmolLM2 (replaces the fixture):
pip install -r requirements.txt
python generate/generate.py           # all examples in examples.py
python generate/generate.py cat-sat   # just one
python generate/embedding_game.py     # real Word Map Game data (needs umap_proj.pkl from generate.py)
python generate/training.py           # real base-vs-Instruct answers for the "How it's trained" tab

# 3. (Optional) Train a tuned lens to denoise the middle layers, then regenerate:
python generate/train_tuned_lens.py   # writes generate/tuned_lens.pt (GPU recommended)
python generate/generate.py           # now also emits logit_lens_tuned + UI toggle
```

There is **no build step** — plain ES modules + CDN libraries. Just serve `web/`.

## How to add a new example

1. **Vet the prompt first**: `python generate/generate.py --preview "your prompt"`
   prints the token split and top-8 predictions without writing anything. Pick
   prompts whose top guess tells a clean story (or a *deliberately* surprising
   one — then add a `note`).
2. Append `{id, title, prompt}` (optional `note` — classroom aside shown in the
   Prediction stage) to `EXAMPLES` in `generate/examples.py`.
3. Run `python generate/generate.py <id>`. It writes `web/examples/<id>.json`
   and rewrites `manifest.json`. Keep prompts **≤ ~12 tokens** (attention data is
   O(layers·heads·N²); short prompts also make clearer "meaning maps").
4. No front-end change needed — the renderer is fully data-driven, including the
   Prediction stage's temperature detents and sampled continuations
   (`temp_chains.bins`) and the still-emitted `attention.spots` (unused by the
   story since the attention section was dropped, kept for the data contract).

## Conventions (important)

- **The renderer must stay data-driven.** Never hardcode layer/head/token counts;
  read them from array lengths (`logit_lens[pos].length`, `attention.weights.length`,
  etc.). This keeps the synthetic fixture (12 layers, 3 heads) and the real model
  (30 layers, 9 heads) both working from the same code.
- **Keep heavy work offline.** At scroll time the JS only decodes/interpolates
  precomputed values — no matrix math beyond a small softmax. Scroll handlers must
  be cheap.
- **Attention is the size-sensitive block** — stored as `ragged_q8` (causal rows
  only, int8). If you change its encoding, update both `generate.py` and
  `decode.js`, and bump the `encoding` tag.
- **The 2D projection is fit once and cached** (`umap_proj.pkl` for UMAP, the
  default; `projection.pkl` for the PCA fallback) so coordinates are stable
  across examples and across the layer trajectory. Deleting a cache re-fits
  and shifts all coordinates — only do it deliberately.
- **Simple mode only.** Copy is written for kids. A former `[Aa]` simplify/detail
  toggle was removed (2026-07): content marked `.aa-detail` (technical asides, raw
  numbers, the Plain/Tuned lens toggle) stays in the markup but is permanently
  hidden by CSS. Keep new copy classroom-appropriate and jargon-light.
- **Accessibility:** respect `prefers-reduced-motion`; the sticky mini-rail is the
  teacher's escape hatch from linear scrolling — keep jump links working.

## Model notes

- SmolLM2-135M: d_model **576**, **30** layers, **9** attention heads (**3** KV
  heads via GQA), vocab 49,152, tied embeddings.
- Logit lens applies the final RMSNorm (`model.model.norm`) then `lm_head` to each
  layer's hidden state. `hidden_states` is length L+1 (index 0 = embeddings).
- We store attention per **query head** (9), since attention probabilities are
  computed per query head even under GQA.

## Status

Built: scrollytelling shell, 4 story stages (Tokens → Embedding → Layers →
Prediction; the former attention section was cut from the story 2026-07 —
attention lives on its own tab + the Architecture lanes, and
web/js/stages/attention.js stays on disk unimported), trajectory overlay on the
Layers meaning map, tuned-lens training + Plain/Tuned toggle (tuned is the
default), curated example set (tokenization demo, pronoun-attention demo,
clean-fact vs surprising-fact pair with `note` asides), data-driven attention
spots (still emitted, story-unused),
`--preview` curation mode, "field guide" visual design (warm graph-paper
theme; Fraunces/Atkinson Hyperlegible/Spline Sans Mono via Google Fonts CDN;
hero + outro sections; per-stage accent colors keyed to the JS token PALETTE;
probability framing line in the Prediction step copy), temperature-binned
sampled continuations in the Prediction stage ("Una parola alla volta", schema
v1.2 `temp_chains`: 5 bins T=0/0.6/0.9/1.3/1.8, 3 seeded sampled variants per
warm bin — 1 for greedy T=0 — of ~12 tokens each, truncated at EOS, newlines
stored as "↵"; the slider snaps to the bins, chips play in one-per-tick with a
reduced-motion instant path, "Un'altra!" cycles variants at the same T; the
bars still re-softmax the stored top-50 logits client-side with an
"everything else" tail bucket that spreads the true leftover mass evenly, so
they match the model's real probabilities at T=1; the v1.1 greedy
`continuation` field, its "Add & predict again" button, and the dice-roll
sampling button were removed with it), architecture-page
"contamination lanes" (attention-rollout ink mixing with a layer scrubber;
sink attention folded into self — see the comment in arch.js), manual controls
alongside scroll (layer slider in Layers, temperature slider in Prediction —
the non-scroll path for projectors and prefers-reduced-motion; scroll re-takes
control on the next scroll event), UMAP meaning map (cosine, fit on the curated
word pool, cached in umap_proj.pkl; tight visible clusters; trajectories are
UMAP transforms of hidden states and land near the predicted word; PCA remains
the no-umap-learn fallback), embedding-stage hero walk (`embedding.hero`: the
neighbors step auto-selects a curated content word, the Layers intro walks back
to the last token; a learner's own strip click pins the selection and stops
the walking), simple-mode-only copy (a former `[Aa]` simplify/detail toggle was
removed 2026-07; `.aa-detail` content — per-stage "for the curious" asides,
token ids, neighbor cosines, the Plain/Tuned lens toggle and label — is still
in the markup/JS but permanently hidden by CSS; web/js/aa.js now only holds
`showLoadError`), "Word map game" standalone tab
(web/embedding-game.html + web/js/game.js: drag a held-out word onto a layer-0
embedding map of curated anchor clusters, scored 0–100 by 2D distance to its
true spot over ~6 rounds; data precomputed by generate/embedding_game.py into
web/embedding_game.json — separate schema v1.0 in SCHEMA.md; `--fixture` mode
runs it without the model; tap-to-place + prefers-reduced-motion fallbacks),
"How it's trained" standalone scrolly tab (web/training.html + web/js/training.js,
"From parrot to partner": one recurring question — "Why is the sky blue?" —
answered across three stages. Pretraining shows the base model *continuing* the
text instead of answering (+ a data-diet bar panel); fine-tuning shows SFT lesson
cards then the Instruct model's fluent-but-myth-containing answer; RL has two
interactive widgets — a "you be the reward model" pick-the-better-answer pair,
and a verifiable-reward math thread with a click-to-reveal thinking trace. Outro
lays the same question's three answers side by side as report cards. Data
precomputed by generate/training.py into web/training.json — separate schema v1.0
in SCHEMA.md; stages 1–2 are real SmolLM2 base-vs-Instruct output (`verbatim`
flag + detail-mode tag), the RL stage is hand-authored in both modes; `--fixture`
runs the whole tab without the model; the page's "real model output" copy is
gated on the data's `verbatim` flags via `.vb-real`/`.vb-demo` spans, and real
mode writes captions matched to the actual greedy outputs), friendly
failure banners on every page (a classic inline script catches file:// —
where ES modules are CORS-blocked — and `showLoadError` in aa.js catches
fetch failures when served), teacher's guide tab (web/teachers.html, static +
printable: lesson pacing, per-tab learning goals / point-at-this / discussion
questions with answer hints, the Paris experiment as a boxed activity, big
misconceptions, student FAQ; print CSS hides the chrome), shared glossary
(web/js/glossary.js: ~20 plain-language terms with `.aa-detail` technical
extras, mounted as a collapsible `<details>` footer on every page and
pre-opened on the teacher's guide so it prints), "Come funziona l'attenzione?"
standalone tab (web/attention.html: a 64-second chaptered animation — sentence
«Ai rigori finisce la partita di calcio», the ambiguous «calcio» vector gets
moved toward its sport sense by attention; React UMD from CDN renders
web/js/anim/engine.js + scene.js, plain JS pre-transpiled offline from a
claude.ai artifact's JSX, fonts/letterbox re-themed to the site; autoplay off,
pauses at chapter ends, overlay is click/tap-to-start), nav order
game → attention → story → architecture → training → teachers (+ external
prompt guide), and the phrase picker moved out of the header into a row below
it (inside #strip-bar on index, its own #picker-bar on architecture).

Not yet built (roughly by classroom value):
- Mobile polish for the attention grid.
- Tuned lens currently trains on wikitext-2; a domain corpus closer to the example
  prompts would denoise the early layers further.
