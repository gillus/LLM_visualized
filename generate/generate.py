"""Offline generator: runs SmolLM2 once per curated example and dumps a
precomputed JSON file the static website renders. NOTHING heavy runs in the
browser — see CLAUDE.md for the architecture and web/SCHEMA.md for the format.

Usage:
    pip install -r requirements.txt
    python generate/generate.py            # all examples in examples.py
    python generate/generate.py cat-sat    # just one (by id)
    python generate/generate.py --preview "Paris is the capital of"
                                           # dry-run a candidate prompt: show its
                                           # token split + top predictions, write
                                           # nothing (for curating examples.py)

The 2D "meaning map" projection is fit ONCE and cached so every example shares
the same coordinates: UMAP (cosine, on the curated word pool) in
generate/umap_proj.pkl, or PCA in generate/projection.pkl when umap-learn
isn't installed. Delete a cache file to re-fit (this shifts ALL coordinates —
do it deliberately).
"""

import json
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import umap            # meaning-map clusters; PCA fallback when missing
except ImportError:
    umap = None

from examples import EXAMPLES
from tuned_lens_model import load_tuned_lens

MODEL = "HuggingFaceTB/SmolLM2-135M"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "web", "examples")
PROJ_CACHE = os.path.join(HERE, "projection.pkl")
UMAP_CACHE = os.path.join(HERE, "umap_proj.pkl")
TUNED_CACHE = os.path.join(HERE, "tuned_lens.pt")

K_LENS = 8       # top-k tokens kept per layer in the logit lens
K_PRED = 50      # top-k kept for the final prediction (enables temperature slider)
CONT_STEPS = 3   # greedy continuation steps ("add the word → predict again" loop)
N_NEIGHBORS = 8  # nearest neighbors per prompt token, in the meaning-map view
CLOUD_SIZE = 400 # background "vocabulary cloud" dots
SEED = 0


def clean_token(t: str):
    """Strip byte-BPE markers for display. Returns (display_text, leading_space)."""
    s = t.replace("Ġ", " ").replace("Ċ", "\n")
    return s.strip(), s.startswith(" ")


def xy2(v):
    """2D point -> compact JSON-safe floats (float32 would serialize as
    1.3660000562667847 and bloat the files)."""
    return [round(float(v[0]), 3), round(float(v[1]), 3)]


@torch.no_grad()
def lens_topk(hs, norm, lm_head, tok, N, tuned=None):
    """Top-k logit-lens guesses per [position][layer]. With `tuned` set, each
    pre-final hidden state is first passed through its trained translator."""
    out = [[] for _ in range(N)]
    n_hs = len(hs)
    for li, layer_vec in enumerate(hs):
        v = layer_vec[0]                       # (N, d_model)
        if li == n_hs - 1:
            h = v                              # final state already post-norm
        elif tuned is not None:
            h = norm(tuned(li, v))             # tuned lens: translate, then final norm
        else:
            h = norm(v)                        # plain lens
        probs = torch.softmax(lm_head(h), dim=-1)
        vals, idx = probs.topk(K_LENS, dim=-1)
        for p in range(N):
            out[p].append([
                {"id": int(idx[p, r]),
                 "text": clean_token(tok.convert_ids_to_tokens(int(idx[p, r])))[0],
                 "prob": round(float(vals[p, r]), 4)}
                for r in range(K_LENS)
            ])
    return out


def topk_pred(logits, tok):
    """Top-K_PRED next-token candidates with logits (for the temperature/dice
    UI) — used for the base prediction and each continuation step."""
    probs = torch.softmax(logits, dim=-1)
    vals, idx = probs.topk(K_PRED)
    return [
        {"id": int(idx[r]),
         "text": clean_token(tok.convert_ids_to_tokens(int(idx[r])))[0],
         "logit": round(float(logits[idx[r]]), 3),
         "prob": round(float(vals[r]), 4)}
        for r in range(K_PRED)
    ]


STOPWORDS = {"the", "a", "an", "of", "to", "in", "on", "at", "is", "was",
             "are", "be", "and", "or", "it", "this", "that", "when", "she",
             "he", "they", "we", "you", "her", "his", "its", "their", "into",
             "with", "for", "called"}


def pick_hero(tokens, neighbors):
    """The embedding stage's default selection. The story's global default is
    the LAST token (where prediction happens), but that's always a function
    word with boring neighbors — the wow-factor neighborhoods belong to
    content words (cat, France, sky). Pick the word-initial content token
    whose nearest neighbor is tightest; fall back to the last token."""
    best = (-1.0, len(tokens) - 1)
    for t in tokens:
        if not any(c.isalpha() for c in t["text"]):
            continue
        if not (t["lead_space"] or t["pos"] == 0):   # skip subword continuations
            continue
        if t["text"].lower() in STOPWORDS:
            continue
        sim = neighbors[str(t["pos"])][0]["sim"]
        if sim > best[0]:
            best = (sim, t["pos"])
    return best[1]


def find_spots(attn, tokens):
    """Curate two interpretable (layer, head) pointers for the attention stage,
    so the scroll steps land on heads that visibly demonstrate something:
      spot 0 — the head where the FINAL token looks back hardest at a
               mid-sentence token (e.g. “she” → “Maria”). Skips position 0,
               which soaks up attention regardless of meaning.
      spot 1 — the most consistent previous-token head (a classic habit).
    The UI's L0H0-style hardcoded steps showed whatever happened to be there —
    usually nothing interpretable."""
    N = len(tokens)
    q = N - 1
    # candidate keys: word-like tokens (punctuation wins raw attention but
    # teaches nothing), and not the immediately previous token — that long-range
    # link is the story (e.g. “she” → “Maria”); spot 1 covers the previous word
    # word-like AND word-initial: continuation fragments ('us' from
    # hipp|opotam|us) read as unrelated words in the caption
    is_word = [any(c.isalpha() for c in t["text"]) and (t["lead_space"] or t["pos"] == 0)
               for t in tokens]
    cand = [k for k in range(1, q - 1) if is_word[k]]
    if not cand:
        cand = [k for k in range(1, q) if is_word[k]] or list(range(1, q))
    # score = weight × lookback distance: a far link (“she” → “Maria”) is the
    # impressive demo; without it nearby words win on raw weight alone
    look = (-1.0, 0, 0, cand[0])
    for L, layer_attn in enumerate(attn):
        a = layer_attn[0]                      # (H, N, N)
        for H in range(a.shape[0]):
            for k in cand:
                w = float(a[H, q, k])
                if w >= 0.15 and w * (q - k) > look[0]:
                    look = (w * (q - k), L, H, k)
    _, lL, lH, lk = look
    prev = (-1.0, 0, 0)
    for L, layer_attn in enumerate(attn):
        a = layer_attn[0]
        for H in range(a.shape[0]):
            if (L, H) == (lL, lH):
                continue
            m = float(sum(a[H, p, p - 1] for p in range(1, N)) / (N - 1))
            if m > prev[0]:
                prev = (m, L, H)
    return [
        {"layer": lL, "head": lH, "focus": lk,
         "why": f"this head connects “{tokens[q]['text']}” back to “{tokens[lk]['text']}”"},
        {"layer": prev[1], "head": prev[2],
         "why": "this head mostly watches the word right before"},
    ]


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32,
        attn_implementation="eager",  # SDPA can't return output_attentions
    )
    model.eval()
    return tok, model


def get_projector(model, tok, words):
    """Fit the shared 2D "meaning map" projection ONCE and cache it.
    Preferred: UMAP (cosine) fit on the curated word pool — gives visibly
    tight clusters (cat/dog/kitten land together), where PCA explains only
    ~14% variance and the clusters are mush. PCA stays as the fallback when
    umap-learn isn't installed. Both caches obey the stability convention:
    refitting shifts every coordinate on the site, so delete a cache file
    only deliberately. Returns (E, projector, projection_metadata)."""
    E = model.get_input_embeddings().weight.detach().numpy()  # (vocab, d_model)
    if umap is not None:
        if os.path.exists(UMAP_CACHE):
            reducer = pickle.load(open(UMAP_CACHE, "rb"))
        else:
            pool = candidate_word_ids(tok, model.config.vocab_size, words)
            reducer = umap.UMAP(n_components=2, metric="cosine",
                                n_neighbors=15, min_dist=0.1, random_state=SEED)
            reducer.fit(E[pool])
            pickle.dump(reducer, open(UMAP_CACHE, "wb"))
            print(f"fit + cached UMAP projection on {len(pool)} curated words")
        meta = {"method": "UMAP", "dims": 2,
                "fit_on": "curated vocab words (cached)",
                "params": {"metric": "cosine", "n_neighbors": 15, "min_dist": 0.1}}
        return E, reducer, meta
    # fallback: linear PCA on the full embedding matrix
    print("umap-learn not installed -> PCA fallback (weaker clusters)")
    if os.path.exists(PROJ_CACHE):
        cached = pickle.load(open(PROJ_CACHE, "rb"))
        # older caches stored (pca, cloud_ids); cloud ids are now curated per run
        pca = cached[0] if isinstance(cached, tuple) else cached
    else:
        pca = PCA(n_components=2).fit(E)
        pickle.dump(pca, open(PROJ_CACHE, "wb"))
        print("fit + cached PCA projection")
    meta = {"method": "PCA", "dims": 2, "fit_on": "vocab (cached)",
            "explained_variance": [round(float(x), 3)
                                   for x in pca.explained_variance_ratio_[:2]]}
    return E, pca, meta


def common_english_words():
    """Lowercased words that occur standalone ≥25× in wikitext-2 (already in the
    HF cache from tuned-lens training). Used to reject BPE word-prefixes like
    'res'/'comp' that pass shape checks but aren't words. Returns None if the
    datasets library is unavailable — callers fall back to shape checks only."""
    try:
        import collections
        import re
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        counts = collections.Counter()
        for line in ds["text"]:
            counts.update(w.lower() for w in re.findall(r"[A-Za-z]+", line))
        return {w for w, c in counts.items() if c >= 25}
    except Exception as e:
        print(f"  (no wikitext word filter: {e})")
        return None


def candidate_word_ids(tok, vocab_size, words=None):
    """Deterministic pool of readable whole-word token ids. A uniform vocab
    sample is mostly BPE fragments ('angar', 'neoc', '\\\\'), which undermines
    the "words with similar meanings sit close" story for a classroom. Keep
    leading-space tokens (whole words under BPE) that are plain alphabetic,
    normally cased, and — when a corpus is available — attested as real
    standalone words. This pool is both the cloud-sampling set and the UMAP
    fit set."""
    raw = tok.convert_ids_to_tokens(list(range(vocab_size)))
    cands = []
    for i, t in enumerate(raw):
        if not t.startswith("Ġ"):
            continue
        w = t[1:]
        if not (w.isascii() and w.isalpha() and 3 <= len(w) <= 12):
            continue
        if not (w.islower() or w.istitle()):
            continue
        if words is not None and w.lower() not in words:
            continue
        cands.append(i)
    return sorted(cands)[: CLOUD_SIZE * 8]   # low ids ~ most frequent words


def curated_cloud_ids(tok, vocab_size, words):
    """Sample CLOUD_SIZE words from the candidate pool for the background of
    the meaning map."""
    cands = candidate_word_ids(tok, vocab_size, words)
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(cands, size=min(CLOUD_SIZE, len(cands)), replace=False))


def export_example(ex, tok, model, E, proj, proj_meta, cloud_ids, tuned=None):
    prompt = ex["prompt"]
    enc = tok(prompt, return_tensors="pt")
    ids = enc.input_ids[0].tolist()
    N = len(ids)

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, output_attentions=True)
    hs = out.hidden_states     # tuple len (L+1), each (1, N, d_model)
    attn = out.attentions      # tuple len L, each (1, H, N, N)
    norm, lm_head = model.model.norm, model.lm_head

    # --- (1) tokens ---
    tokens = []
    for pos, i in enumerate(ids):
        txt, lead = clean_token(tok.convert_ids_to_tokens(i))
        tokens.append({"pos": pos, "id": int(i), "text": txt, "lead_space": lead})

    # --- (2) embedding: prompt tokens, vocab cloud, neighbors ---
    # one batched proj.transform per group: UMAP's transform has real per-call
    # overhead, so per-point calls would dominate generation time
    tok_xy = proj.transform(E[ids])
    emb_tokens = [{"pos": p, "xy": xy2(tok_xy[p])} for p in range(N)]
    cloud_xy = proj.transform(E[np.asarray(cloud_ids)])
    cloud = [
        {"id": int(c), "text": clean_token(tok.convert_ids_to_tokens(int(c)))[0],
         "xy": xy2(cloud_xy[ci])}
        for ci, c in enumerate(cloud_ids)
    ]
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    nbr_tops = []
    for p, i in enumerate(ids):
        sims = En @ En[i]
        sims[i] = -1.0
        nbr_tops.append((sims, np.argsort(-sims)[:N_NEIGHBORS]))
    flat_ids = [int(j) for _, top in nbr_tops for j in top]
    flat_xy = proj.transform(E[flat_ids])
    neighbors, fi = {}, 0
    for p, (sims, top) in enumerate(nbr_tops):
        neighbors[str(p)] = []
        for j in top:
            neighbors[str(p)].append(
                {"id": int(j), "text": clean_token(tok.convert_ids_to_tokens(int(j)))[0],
                 "sim": round(float(sims[j]), 3),
                 "xy": xy2(flat_xy[fi])})
            fi += 1

    # --- (3) logit lens (all positions, all layers) + 2D trajectory ---
    # hs has L+1 entries: [0]=embeddings, [1..L-1]=intermediate (pre-norm),
    # [L]=final hidden state which HF has ALREADY passed through the final norm.
    # The trajectory is lens-independent: it's just the hidden state projected.
    # (Under UMAP-cosine this still tells the right story: the final state's
    # DIRECTION aligns with the predicted word's tied embedding, so the path
    # ends near that word's spot on the map.)
    n_hs = len(hs)
    all_h = np.concatenate([h[0].detach().numpy() for h in hs], axis=0)
    all_xy = proj.transform(all_h).reshape(n_hs, N, 2)
    # UMAP's transform snaps mid-stack (out-of-distribution) states to whichever
    # cluster is nearest, which draws an unreadable zigzag (~9× the direct
    # distance; smoothing in 576-D first doesn't help — the snapping happens in
    # the projection). So smooth the DRAWN 2D path with an EMA. Endpoints stay
    # exact: layer 0 is the word's true spot, the last layer is the true final
    # state (which lands near the predicted word under cosine).
    TRAJ_EMA = 0.2
    for p in range(N):
        for li in range(1, n_hs - 1):
            all_xy[li, p] = TRAJ_EMA * all_xy[li, p] + (1 - TRAJ_EMA) * all_xy[li - 1, p]
    traj = [[xy2(all_xy[li, p]) for li in range(n_hs)] for p in range(N)]

    lens = lens_topk(hs, norm, lm_head, tok, N)                         # plain
    lens_tuned = (lens_topk(hs, norm, lm_head, tok, N, tuned)           # tuned (optional)
                  if tuned is not None else None)

    # --- (4) attention -> ragged + int8 (causal: row q keeps only keys 0..q) ---
    aw = []
    for layer_attn in attn:                    # (1, H, N, N)
        a = layer_attn[0]
        heads = []
        for h in range(a.shape[0]):
            rows = [[int(round(float(a[h, q, k]) * 255)) for k in range(q + 1)]
                    for q in range(N)]
            heads.append(rows)
        aw.append(heads)

    # --- (5) prediction at last position (store logits for temperature slider) ---
    # Use the model's own logits (ground truth) — no manual norm, no double-counting.
    final = out.logits[0, -1]                  # (vocab,)
    pred = topk_pred(final, tok)

    # --- (6) autoregressive continuation: greedily append the top word and
    # re-run, CONT_STEPS times. Only the next-word distribution is kept per
    # step (embedding/layers/attention are NOT re-run — see SCHEMA.md). This
    # powers the story's "add the word → predict again" loop.
    chain = []
    chain_ids = list(ids)
    last_logits = final
    with torch.no_grad():
        for _ in range(CONT_STEPS):
            nid = int(last_logits.argmax())
            txt, lead = clean_token(tok.convert_ids_to_tokens(nid))
            chain_ids.append(nid)
            out_c = model(input_ids=torch.tensor([chain_ids]))
            last_logits = out_c.logits[0, -1]
            chain.append({"token": {"id": nid, "text": txt, "lead_space": lead},
                          "top": topk_pred(last_logits, tok)})

    doc = {
        "schema_version": "1.1",
        "id": ex["id"], "title": ex["title"], "prompt": prompt,
        "model": {
            "name": MODEL,
            "d_model": model.config.hidden_size,
            "n_layers": model.config.num_hidden_layers,
            "n_heads": model.config.num_attention_heads,
            "n_kv_heads": getattr(model.config, "num_key_value_heads",
                                  model.config.num_attention_heads),
            "vocab_size": model.config.vocab_size,
            "tied_embeddings": getattr(model.config, "tie_word_embeddings", True),
        },
        "tokens": tokens,
        "projection": proj_meta,
        "embedding": {"tokens": emb_tokens, "vocab_cloud": cloud,
                      "neighbors": neighbors, "hero": pick_hero(tokens, neighbors)},
        "logit_lens": lens,
        "trajectory": traj,
        "attention": {"encoding": "ragged_q8", "scale": round(1 / 255, 6),
                      "weights": aw, "spots": find_spots(attn, tokens)},
        "prediction": {"position": N - 1, "top": pred,
                       "temperature_note": "UI re-softmaxes logit/T over this set"},
        "continuation": chain,
    }

    if lens_tuned is not None:
        doc["logit_lens_tuned"] = lens_tuned
    if ex.get("note"):
        doc["note"] = ex["note"]

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{ex['id']}.json")
    json.dump(doc, open(path, "w"), separators=(",", ":"))
    print(f"  wrote {path}  ({os.path.getsize(path) // 1024} KB, {N} tokens)")


def write_manifest(examples):
    manifest = [{"id": e["id"], "title": e["title"], "prompt": e["prompt"]}
                for e in examples]
    path = os.path.join(OUT_DIR, "manifest.json")
    json.dump(manifest, open(path, "w"), indent=2)
    print(f"updated {path}")


@torch.no_grad()
def preview(prompts):
    """Dry-run candidate prompts: print the token split and the model's top
    predictions so a curator can judge whether the example tells a clean story
    before adding it to examples.py. Writes nothing."""
    tok, model = load_model()
    for prompt in prompts:
        enc = tok(prompt, return_tensors="pt")
        ids = enc.input_ids[0].tolist()
        pieces = []
        for i in ids:
            txt, lead = clean_token(tok.convert_ids_to_tokens(i))
            pieces.append(("␣" if lead else "") + (txt or "␣"))
        out = model(**enc)
        probs = torch.softmax(out.logits[0, -1], dim=-1)
        vals, idx = probs.topk(8)
        print(f"\nprompt: {prompt!r}  ({len(ids)} tokens)")
        print("tokens:", " | ".join(pieces))
        print("top-8: ", ", ".join(
            f"{clean_token(tok.convert_ids_to_tokens(int(i)))[0] or '␣'} {float(v):.1%}"
            for v, i in zip(vals, idx)))


def main():
    args = sys.argv[1:]
    if args and args[0] == "--preview":
        if not args[1:]:
            print('usage: generate.py --preview "prompt one" ["prompt two" ...]')
            return
        preview(args[1:])
        return
    wanted = args
    todo = [e for e in EXAMPLES if not wanted or e["id"] in wanted]
    if not todo:
        print(f"no matching examples for {wanted}")
        return
    tok, model = load_model()
    words = common_english_words()   # shared by the cloud sampler + UMAP fit
    E, proj, proj_meta = get_projector(model, tok, words)
    cloud_ids = curated_cloud_ids(tok, model.config.vocab_size, words)
    print(f"projection: {proj_meta['method']}")

    tuned = None
    if os.path.exists(TUNED_CACHE):
        tuned, ck = load_tuned_lens(TUNED_CACHE)
        print(f"loaded tuned lens ({ck['n_layers']} layers) -> emitting logit_lens_tuned")
    else:
        print("no tuned_lens.pt found -> plain lens only "
              "(run train_tuned_lens.py to add a tuned lens)")

    for ex in todo:
        print(f"generating {ex['id']} …")
        export_example(ex, tok, model, E, proj, proj_meta, cloud_ids, tuned)
    write_manifest(EXAMPLES)   # manifest always lists the full curated set
    print("done.")


if __name__ == "__main__":
    main()
