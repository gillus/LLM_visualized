# Example JSON schema (v1.2)

One file per example at `web/examples/<id>.json`. This is the contract between
the offline generator (`generate/generate.py`) and the static renderer (`web/js`).
The renderer reads **all sizes from the data** — never assume layer/head/token
counts.

```jsonc
{
  "schema_version": "1.0",
  "id": "cat-sat",
  "title": "The cat sat on the…",          // dropdown label
  "prompt": "The cat sat on the",

  "model": {                                // metadata for captions
    "name": "HuggingFaceTB/SmolLM2-135M",
    "d_model": 576, "n_layers": 30, "n_heads": 9, "n_kv_heads": 3,
    "vocab_size": 49152, "tied_embeddings": true
  },

  // ① TOKENS
  "tokens": [
    {"pos": 0, "id": 504, "text": "The", "lead_space": false}
    // text = display-cleaned (Ġ stripped); lead_space = render a gap before it
  ],

  // ② EMBEDDING (meaning map) — 2D projection, fit ONCE & shared across
  // examples (and across the layer trajectory). method = "UMAP" (cosine, fit
  // on the curated word pool, cached in generate/umap_proj.pkl — tight visible
  // clusters) or "PCA" (fallback when umap-learn is missing; ~14% variance).
  // The renderer is metadata-agnostic: it only reads xy values.
  "projection": {"method": "UMAP", "dims": 2,
                 "params": {"metric": "cosine", "n_neighbors": 15, "min_dist": 0.1}},
  "embedding": {
    "tokens": [{"pos": 0, "xy": [-1.42, 0.88]}],          // each prompt token
    "vocab_cloud": [{"id": 7411, "text": "dog", "xy": [1.1, 0.9]}],  // bg context
    // neighbors: cosine top-k in full d_model space; xy = same 2D projection
    // so the renderer can place them on the map (older files may lack xy)
    "neighbors": {"1": [{"id": 7411, "text": "dog", "sim": 0.71, "xy": [1.1, 0.9]}]},
    // hero: OPTIONAL — the content token the Embedding stage auto-selects on
    // its "neighbors" step (the last token is always a boring function word).
    // Curated offline: tightest-neighborhood content word. The walk reverts
    // to the last token when the Layers stage begins, unless the learner has
    // clicked a token themselves.
    "hero": 1
  },

  // ③ LAYERS — logit lens, indexed [pos][layer][rank]; layer 0 = embeddings
  "logit_lens": [
    [ [ {"id": 13, "text": "floor", "prob": 0.04} ] ]     // pos → layer → top-k
  ],
  "logit_lens_tuned": [ /* same shape */ ], // OPTIONAL — present iff a tuned lens
                                            // was trained (generate/tuned_lens.pt)
  "trajectory": [ [ [-1.42, 0.88] ] ],   // OPTIONAL [pos][layer]=xy, same proj space

  // ④ ATTENTION — ragged + int8. weights[layer][head][query] = [key0..keyQ]
  "attention": {
    "encoding": "ragged_q8",             // causal rows only; multiply by scale
    "scale": 0.003922,                   // int8 → float (1/255)
    "weights": [ [ [ [255], [120,135] ] ] ],
    // curated interpretable heads — the "spot<n>" scroll steps jump to these.
    // spots[0]: where the final token looks back hardest ("she" → "Maria");
    // spots[1]: a previous-token head. `why` is shown as the panel caption
    // (only while the final token is selected, if `focus` is set).
    "spots": [ {"layer": 3, "head": 8, "focus": 1, "why": "this head connects “she” back to “Maria”"} ]
  },

  // ⑤ PREDICTION — final next-token dist; store LOGITS so the UI can re-softmax
  "prediction": {
    "position": 4,                       // index of the token we predict after
    "top": [ {"id": 2113, "text": "mat", "logit": 14.2, "prob": 0.41} ]  // ~top 50
  },

  // OPTIONAL — precomputed sampled continuations, one set per temperature bin
  // (replaces the v1.1 `continuation` greedy chain). bins are ordered cold→hot
  // and the UI reads bins.length for the slider detents — never assume 5.
  // T=0 stores ONE variant (greedy is deterministic; the UI says "always the
  // same"); warm bins store a few independently sampled variants the learner
  // can cycle through. Token dicts use the same display convention as `tokens`
  // (text + lead_space); chains stop early at EOS and newline tokens are
  // stored as a visible "↵". The UI hides the chain panel when absent.
  "temp_chains": {
    "steps": 12,                        // requested length (chains may be shorter)
    "bins": [
      {"t": 0.0, "variants": [ [ {"text": "mat", "lead_space": true} ] ]},
      {"t": 0.9, "variants": [ [/* 12 tokens */], [/*…*/], [/*…*/] ]}
    ]
  },

  // OPTIONAL — classroom aside rendered in the Prediction stage; use when the
  // top guess is a teachable surprise (set per example in generate/examples.py)
  "note": "Surprised? “Paris” is only the second guess! …"
}
```

## Notes

- **`logit_lens` is per-position** so a learner can click any token in the strip
  and scrub its prediction forming across layers. `logit_lens[pos].length` =
  `n_layers + 1`.
- **`logit_lens_tuned`** (optional) has the identical shape and indexing. When
  present, the Layers stage shows a Plain/Tuned toggle. The plain lens applies the
  frozen output head at every layer (noisy mid-stack on a tiny model); the tuned
  lens first passes each layer through a trained per-layer affine translator, so
  the middle layers read out cleanly. The final layer is identical in both (it's
  the model's real output). Generated by `generate/train_tuned_lens.py`.
- **`prediction.top` stores logits**, not just probs — the temperature slider does
  `softmax(logit / T)` over this set client-side (an approximation over top-50,
  invisible at classroom scale).
- **`attention.encoding`**: `ragged_q8` (default) or `dense_f32`. `decode.js`
  expands ragged rows to a dense `[q][k]` float matrix, zero-padding `k > q`.
- **`temp_chains`** is display-only: the other stages (embedding/layers/
  attention) are NOT re-run for the continuation tokens. Chains are sampled
  offline with `torch.multinomial(softmax(logit / T))` using a fixed seed per
  (example, bin, variant) — `zlib.crc32(f"{id}/{bin}/{variant}")` — so
  regenerating writes byte-identical files. The Prediction stage's temperature
  slider snaps to these bins; the probability bars still re-softmax the stored
  `prediction.top` logits client-side at the bin's T.
- **`trajectory`** uses the same projection as `embedding`, so a token's hidden
  state can be drawn drifting across the meaning map as layers progress. Under
  UMAP-cosine this is a `transform` of out-of-distribution points — defensible
  here because the final hidden state's direction aligns with the predicted
  word's tied embedding, so the path visibly ends near that word. The stored
  2D path is EMA-smoothed (α=0.2) for readability — raw UMAP transforms of
  mid-stack states zigzag ~9× the direct distance; the first and last points
  are exact. Optional.

## Word Map Game — `web/embedding_game.json` (separate file, schema v1.0)

The "Word map game" tab uses its own precomputed file, NOT a per-example one.
Generated by `generate/embedding_game.py` (real model) or that script's
`--fixture` mode (synthetic, no model). A curated word pool is placed on the
SAME meaning map the story uses (layer-0 token embeddings → cached UMAP), so
coordinates are comparable across both pages. The browser scores a guess purely
by 2D distance to the stored `xy` — no model, no inverse projection.

```jsonc
{
  "schema_version": "1.0",
  "projection": {"method": "UMAP", "dims": 2, "fit_on": "layer-0 token embeddings"},
  "categories": [{"name": "animals", "color": "#d97706"}],   // dot/label colour per cluster
  "anchors": [                                               // pre-placed, labelled landmarks
    {"word": "dog", "cat": "animals", "xy": [1.2, -0.4]}
  ],
  "rounds": [                                                // held-out mystery words to drag
    {"word": "wolf", "cat": "animals", "xy": [1.0, -0.3],
     "neighbors": [{"word": "dog", "sim": 0.71}],            // full-space cosine, for feedback
     "note": null}                                           // optional classroom aside
  ]
}
```

The renderer (`web/js/game.js`) stays data-driven: categories, colours, anchor
count and round count all come from this file.

## Training tour — `web/training.json` (separate file, schema v1.0)

The "How it's trained" tab (`web/training.html` + `web/js/training.js`,
"From parrot to partner") walks the three training stages on one recurring
question. Generated by `generate/training.py` (real **base vs Instruct**
completions for stages 1–2) or its `--fixture` mode (fully hand-authored). The
RL stage is hand-authored in BOTH modes — RLHF can't be run for a demo, and
the idea is the point. `verbatim` flags whether an answer is real model output
(lightly trimmed) or an illustrative stand-in; the renderer shows a matching
tag in detail mode.

```jsonc
{
  "schema_version": "1.0",
  "model": {"base": "…-135M", "instruct": "…-135M-Instruct"},
  "prompt": "Why is the sky blue?",              // the question asked at every stage
  "data_mix": [{"label": "Web pages", "pct": 58}],  // pretraining diet bars (illustrative)

  "pretrain": {"answer": "…the base model CONTINUES the text…",
               "verbatim": true, "caption": "…why it doesn't answer…"},
  "finetune": {"examples": [{"instruction": "…", "response": "…"}],  // SFT lesson cards
               "answer": "…the Instruct model ANSWERS…",
               "verbatim": true, "caption": "…fluent but can be wrong…"},
  "rl": {
    "preference": {                              // (a) "you be the reward model"
      "question": "Why is the sky blue?",
      "candidates": [{"text": "…", "better": false, "tag": "sounds nice, but wrong"},
                     {"text": "…", "better": true,  "tag": "clear and correct"}],
      "reward_up": "…", "reward_down": "…", "reveal": "…which trainers prefer & why…"
    },
    "verifiable": {                              // (b) verifiable reward → thinking
      "problem": "…", "naive_answer": "$5", "naive_note": "…",
      "thinking": ["1 notebook = $3", "…"], "answer": "$11", "reward_note": "…"
    }
  },
  "report_cards": [                              // outro: same question, three verdicts
    {"stage": "Pretrained", "kind": "parrot", "verdict": "…"}  // kind ∈ parrot|helper|expert
  ]
}
```

The renderer pulls the parrot/helper answers and the *better* preference
candidate into the three outro report cards; `kind` keys the accent colour.

## Approx sizes (N=8 tokens, 30 layers, 9 heads, k=8)

| block | with tricks |
|---|---|
| attention (ragged + int8) | ~35 KB |
| logit lens | ~40–90 KB |
| embedding + cloud | ~12 KB |
| **total raw** | ~140 KB (≈ 40–60 KB gzipped) |

Attention scales O(layers·heads·N²) — keep prompts short.
