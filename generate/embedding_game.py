"""Offline generator for the "Word Map Game" tab.

Dumps web/embedding_game.json: a curated pool of words, each placed on the SAME
2D "meaning map" the story page uses (layer-0 token embeddings -> cached UMAP),
split into visible labelled ANCHORS (the landmarks) and held-out ROUNDS (the
words the learner drags). The browser scores a guess purely by 2D distance to
the stored true position, so all the model work stays offline.

    python generate/embedding_game.py            # real SmolLM2 data
    python generate/embedding_game.py --fixture   # synthetic, no model needed

The real run reuses generate/umap_proj.pkl (the cached projection fit by
generate.py). Run generate.py at least once first so that cache exists; deleting
it re-fits and shifts every coordinate on both pages.
"""
import argparse
import json
import math
import os
import pickle
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "web", "embedding_game.json")
UMAP_CACHE = os.path.join(HERE, "umap_proj.pkl")
PROJ_CACHE = os.path.join(HERE, "projection.pkl")
MODEL = "HuggingFaceTB/SmolLM2-135M"

# --- the curated word pool: clear categories make legible clusters ----------
# Each category gets a colour the front-end reuses for its dots + label.
CATEGORIES = [
    {"name": "animals",  "color": "#d97706"},
    {"name": "cities",   "color": "#2563eb"},
    {"name": "emotions", "color": "#db2777"},
    {"name": "colors",   "color": "#7c3aed"},
    {"name": "food",     "color": "#16a34a"},
    {"name": "family",   "color": "#0891b2"},
]

# All words, by category. The ROUNDS list below pulls a few of these out to be
# the held-out mystery words; everything else is drawn as a labelled anchor.
WORDS = {
    "animals":  ["dog", "cat", "horse", "lion", "tiger", "rabbit", "mouse",
                 "bear", "wolf", "fox"],
    "cities":   ["Paris", "London", "Tokyo", "Rome", "Berlin", "Moscow",
                 "Madrid", "Cairo"],
    "emotions": ["happy", "sad", "angry", "afraid", "love", "fear", "calm",
                 "proud", "lonely"],
    "colors":   ["red", "blue", "green", "yellow", "purple", "pink", "black",
                 "white", "orange"],
    "food":     ["bread", "cheese", "apple", "banana", "rice", "soup", "meat",
                 "fish", "cake"],
    "family":   ["mother", "father", "sister", "brother", "aunt", "son",
                 "daughter", "uncle"],
}

# The mystery words (≈6 rounds), one per category. `orange` is a deliberate
# between-clusters teaser: it's a colour the model also ties to fruit.
ROUNDS = [
    {"word": "wolf",   "cat": "animals",  "note": None},
    {"word": "Madrid", "cat": "cities",   "note": None},
    {"word": "proud",  "cat": "emotions", "note": None},
    {"word": "orange", "cat": "colors",
     "note": "In inglese “orange” è anche un frutto — guarda se scivola verso i cibi."},
    {"word": "cake",   "cat": "food",     "note": None},
    {"word": "uncle",  "cat": "family",   "note": None},
]
ROUND_WORDS = {r["word"] for r in ROUNDS}
K_NEIGHBORS = 4


def xy(v):
    return [round(float(v[0]), 3), round(float(v[1]), 3)]


# ---------------------------------------------------------------------------
# REAL: SmolLM2 layer-0 embeddings -> cached UMAP
# ---------------------------------------------------------------------------
def build_real():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.exists(UMAP_CACHE) and not os.path.exists(PROJ_CACHE):
        raise SystemExit(
            "No projection cache found. Run `python generate/generate.py` once "
            "first so the shared meaning-map projection exists, then re-run this.")
    cache = UMAP_CACHE if os.path.exists(UMAP_CACHE) else PROJ_CACHE
    method = "UMAP" if cache is UMAP_CACHE else "PCA"
    proj = pickle.load(open(cache, "rb"))
    print(f"loaded {method} projection from {os.path.basename(cache)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    E = model.get_input_embeddings().weight.detach().cpu().numpy()  # (vocab, d)

    # resolve each word -> a single whole-word (leading-space) token id
    ids, kept = {}, []
    for cat, words in WORDS.items():
        for w in words:
            tid = tok.convert_tokens_to_ids("Ġ" + w)
            if tid is None or tid == tok.unk_token_id:
                print(f"  skip '{w}' (not a single token)")
                continue
            ids[w] = tid
            kept.append((w, cat))

    vecs = np.stack([E[ids[w]] for w, _ in kept])
    coords = proj.transform(vecs)
    coord_of = {w: coords[i] for i, (w, _) in enumerate(kept)}

    # full-space cosine neighbours within the pool (for "closest to…" feedback)
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    words_kept = [w for w, _ in kept]

    def neighbors_of(w):
        i = words_kept.index(w)
        order = np.argsort(-sim[i])
        out = []
        for j in order:
            if words_kept[j] == w:
                continue
            out.append({"word": words_kept[j], "sim": round(float(sim[i, j]), 3)})
            if len(out) >= K_NEIGHBORS:
                break
        return out

    anchors = [{"word": w, "cat": c, "xy": xy(coord_of[w])}
               for w, c in kept if w not in ROUND_WORDS]
    rounds = []
    for r in ROUNDS:
        w = r["word"]
        if w not in coord_of:
            print(f"  WARNING: round word '{w}' missing; dropping it")
            continue
        rounds.append({"word": w, "cat": r["cat"], "xy": xy(coord_of[w]),
                       "neighbors": neighbors_of(w), "note": r["note"]})

    meta = {"method": method, "dims": 2, "fit_on": "layer-0 token embeddings"}
    return meta, anchors, rounds


# ---------------------------------------------------------------------------
# FIXTURE: hand-placed cluster centres + jitter (no model)
# ---------------------------------------------------------------------------
def build_fixture():
    random.seed(11)
    centers = {  # rough cluster centres on a tidy 2D plane
        "animals":  (-3.2,  2.4),
        "cities":   ( 3.0,  2.8),
        "emotions": (-3.6, -2.2),
        "colors":   ( 0.2,  3.6),
        "food":     ( 3.4, -1.8),
        "family":   (-0.6, -3.4),
    }

    def place(cat, pull=0.0, toward=None):
        cx, cy = centers[cat]
        px, py = (cx, cy)
        if toward is not None:  # nudge a word partway to another cluster
            tx, ty = centers[toward]
            px, py = cx + (tx - cx) * pull, cy + (ty - cy) * pull
        return (round(px + random.uniform(-0.7, 0.7), 3),
                round(py + random.uniform(-0.7, 0.7), 3))

    coord_of, cat_of = {}, {}
    for cat, words in WORDS.items():
        for w in words:
            # the "orange" teaser sits halfway to the food cluster
            if w == "orange":
                coord_of[w] = place("colors", pull=0.65, toward="food")
            else:
                coord_of[w] = place(cat)
            cat_of[w] = cat

    def neighbors_of(w):
        wx, wy = coord_of[w]
        others = [o for o in coord_of if o != w]
        others.sort(key=lambda o: (coord_of[o][0] - wx) ** 2 + (coord_of[o][1] - wy) ** 2)
        out = []
        for j, o in enumerate(others[:K_NEIGHBORS]):
            d = math.hypot(coord_of[o][0] - wx, coord_of[o][1] - wy)
            out.append({"word": o, "sim": round(max(0.2, 0.95 - 0.12 * d), 3)})
        return out

    anchors = [{"word": w, "cat": cat_of[w], "xy": list(coord_of[w])}
               for w in coord_of if w not in ROUND_WORDS]
    rounds = [{"word": r["word"], "cat": r["cat"], "xy": list(coord_of[r["word"]]),
               "neighbors": neighbors_of(r["word"]), "note": r["note"]}
              for r in ROUNDS]
    meta = {"method": "SYNTHETIC", "dims": 2, "fit_on": "hand-placed clusters"}
    return meta, anchors, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true",
                    help="write synthetic data (no model download)")
    args = ap.parse_args()

    meta, anchors, rounds = build_fixture() if args.fixture else build_real()
    doc = {
        "schema_version": "1.0",
        "projection": meta,
        "categories": CATEGORIES,
        "anchors": anchors,
        "rounds": rounds,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w"), separators=(",", ":"))
    print(f"wrote {os.path.relpath(OUT)}  "
          f"({len(anchors)} anchors, {len(rounds)} rounds, {meta['method']})")


if __name__ == "__main__":
    main()
