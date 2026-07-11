"""Create a SYNTHETIC example file so the website runs without downloading the
model. Numbers are fabricated (deterministic) but structurally identical to what
generate.py produces. Run the real generate.py later to replace it with true
SmolLM2 data.

    python generate/fixture.py
"""
import json
import math
import os
import random

random.seed(7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "web", "examples")

PROMPT_TOKENS = [("The", False), ("cat", True), ("sat", True), ("on", True), ("the", True)]
TARGETS = ["floor", "wall", "step", "mat", "mat"]  # what each position "wants" to predict
N = len(PROMPT_TOKENS)
L = 12                      # attention layers; logit lens has L+1 entries
H = 3                       # heads
K_LENS = 6
CLOUD = ("dog cat kitten puppy animal pet sat sit ran jump floor mat rug ground "
         "couch table chair the a an on in under house home run walk red blue sky").split()
PRED = [("mat", 5.2), ("floor", 4.6), ("rug", 4.1), ("ground", 3.4),
        ("couch", 2.9), ("table", 2.5), ("carpet", 2.0), ("bed", 1.6)]


def jitter(base, amt):
    return [round(base[0] + random.uniform(-amt, amt), 3),
            round(base[1] + random.uniform(-amt, amt), 3)]


# token meaning-map anchors (so neighbors cluster believably)
anchors = {w: (random.uniform(-3, 3), random.uniform(-3, 3)) for w in set(CLOUD + TARGETS)}
for w, _ in PROMPT_TOKENS:
    anchors.setdefault(w, (random.uniform(-3, 3), random.uniform(-3, 3)))

tokens = [{"pos": i, "id": 300 + i * 137, "text": t, "lead_space": ls}
          for i, (t, ls) in enumerate(PROMPT_TOKENS)]

emb_tokens = [{"pos": i, "xy": jitter(anchors[t["text"]], 0.05)} for i, t in enumerate(tokens)]
vocab_cloud = [{"id": 1000 + i, "text": w, "xy": jitter(anchors[w], 0.4)}
               for i, w in enumerate(CLOUD)]
neighbors = {}
for i, t in enumerate(tokens):
    pool = sorted(CLOUD, key=lambda w: (anchors[w][0] - anchors[t["text"]][0]) ** 2
                  + (anchors[w][1] - anchors[t["text"]][1]) ** 2)
    neighbors[str(i)] = [{"id": 1000 + CLOUD.index(w), "text": w,
                          "sim": round(0.9 - 0.07 * j, 3),
                          "xy": vocab_cloud[CLOUD.index(w)]["xy"]}
                         for j, w in enumerate(pool[:8])]

# logit lens: early layers random, converging to TARGETS[pos] by the last layer
DISTRACT = ["the", "a", ",", "and", "is", "to", "of", "in", "it", "that"]
logit_lens, trajectory = [], []
for p in range(N):
    per_layer, traj = [], []
    target = TARGETS[p]
    start = anchors[tokens[p]["text"]]
    for layer in range(L + 1):
        conf = (layer / L) ** 2  # confidence grows with depth
        words = [target] + random.sample(DISTRACT, K_LENS - 1)
        random.shuffle(words)
        raw = []
        for w in words:
            base = (0.55 * conf + 0.4) if w == target else random.uniform(0.02, 0.3)
            raw.append(max(base, 0.001))
        s = sum(raw)
        guesses = sorted(
            [{"id": 2000 + hash(w) % 5000, "text": w, "prob": round(v / s, 4)}
             for w, v in zip(words, raw)],
            key=lambda d: -d["prob"])
        per_layer.append(guesses)
        # trajectory drifts from token anchor toward target anchor
        tx, ty = anchors[tokens[p]["text"]]
        gx, gy = anchors[target]
        traj.append([round(tx + (gx - tx) * conf, 3), round(ty + (gy - ty) * conf, 3)])
    logit_lens.append(per_layer)
    trajectory.append(traj)

# attention (ragged int8): later layers attend back to "cat"/"sat" more
weights = []
for layer in range(L):
    heads = []
    for h in range(H):
        rows = []
        for q in range(N):
            r = []
            for k in range(q + 1):
                w = random.uniform(0.2, 1.0)
                if layer > L // 2 and k in (1, 2):  # bias to subject/verb late
                    w += 1.2
                if k == q:
                    w += 0.5
                r.append(w)
            tot = sum(r)
            rows.append([int(round(v / tot * 255)) for v in r])
        heads.append(rows)
    weights.append(heads)

def dist(pairs, base_id):
    """(word, logit) list -> prediction-format dicts with softmaxed probs."""
    rows = [{"id": base_id + i, "text": w, "logit": lg, "prob": 0.0}
            for i, (w, lg) in enumerate(pairs)]
    mx = max(r["logit"] for r in rows)
    exs = [math.exp(r["logit"] - mx) for r in rows]
    ssum = sum(exs)
    for r, e in zip(rows, exs):
        r["prob"] = round(e / ssum, 4)
    return rows


pred = dist(PRED, 2000)

# synthetic greedy continuation: "mat" -> "." -> "The" (token = previous top-1)
CONT = [
    ("mat", True, [(".", 5.0), ("and", 4.2), (",", 3.8), ("in", 3.0),
                   ("again", 2.4), ("all", 2.0), ("to", 1.7), ("happily", 1.4)]),
    (".", False, [("The", 4.8), ("It", 4.1), ("Then", 3.5), ("A", 3.0),
                  ("She", 2.3), ("And", 2.0), ("He", 1.8), ("One", 1.5)]),
    ("The", True, [("cat", 5.1), ("dog", 4.0), ("sun", 3.2), ("next", 2.6),
                   ("mat", 2.2), ("end", 1.9), ("door", 1.6), ("day", 1.4)]),
]
continuation = [
    {"token": {"id": 2500 + i, "text": w, "lead_space": lead},
     "top": dist(pairs, 2600 + i * 100)}
    for i, (w, lead, pairs) in enumerate(CONT)
]

doc = {
    "schema_version": "1.1", "id": "cat-sat", "title": "The cat sat on the…",
    "prompt": "The cat sat on the",
    "model": {"name": "SYNTHETIC-FIXTURE", "d_model": 576, "n_layers": L,
              "n_heads": H, "n_kv_heads": 1, "vocab_size": 49152, "tied_embeddings": True},
    "tokens": tokens,
    "projection": {"method": "PCA", "dims": 2, "fit_on": "synthetic",
                   "explained_variance": [0.21, 0.13]},
    "embedding": {"tokens": emb_tokens, "vocab_cloud": vocab_cloud,
                  "neighbors": neighbors, "hero": 1},   # "cat"
    "logit_lens": logit_lens, "trajectory": trajectory,
    "attention": {"encoding": "ragged_q8", "scale": round(1 / 255, 6), "weights": weights,
                  # synthetic spots: late layers are biased toward k=1/2 ("cat"/"sat")
                  "spots": [
                      {"layer": L - 2, "head": 0, "focus": 1,
                       "why": "this head connects “the” back to “cat”"},
                      {"layer": 0, "head": 1,
                       "why": "this head mostly watches the word right before"},
                  ]},
    "prediction": {"position": N - 1, "top": pred,
                   "temperature_note": "UI re-softmaxes logit/T over this set"},
    "continuation": continuation,
}

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(doc, open(os.path.join(OUT_DIR, "cat-sat.json"), "w"), separators=(",", ":"))
json.dump([{"id": "cat-sat", "title": "The cat sat on the… (synthetic demo)",
            "prompt": "The cat sat on the"}],
          open(os.path.join(OUT_DIR, "manifest.json"), "w"), indent=2)
print("wrote synthetic web/examples/cat-sat.json + manifest.json")
