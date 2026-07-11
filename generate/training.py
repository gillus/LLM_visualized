"""Offline generator for the "From parrot to partner" tab.

Dumps web/training.json: the content for a scrollytelling page about the three
schools a language model goes through —

    1. PRETRAINING   reads the internet, becomes an autocomplete tool
    2. FINE-TUNING   shown example conversations, becomes an assistant
    3. RL            graded on its answers, becomes more accurate + learns to think

The spine is one recurring question ("Why is the sky blue?") answered three
times so the learner SEES the model grow up. Only the first two answers come
from a real model — base vs instruct — because that contrast is honest and
on-brand (same SmolLM2 family the rest of the site uses). The RL stage is
hand-authored: you can't cheaply run RLHF for a classroom demo, and the point
there is the *idea* of a reward signal, which the authored examples carry.

    python generate/training.py            # real base-vs-instruct completions
    python generate/training.py --fixture   # fully hand-authored, no model

Real mode only swaps in `pretrain.answer` (base model continuing the prompt)
and `finetune.answer` (instruct model answering it). Everything else — the SFT
example cards, the reward-model pair, the math thread — is the AUTHORED content
below in both modes. A 135M model's raw text is rough, so each generated answer
is lightly trimmed (kept whole sentences, capped length) and tagged so the UI
can say whether it's verbatim.
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "web", "training.json")
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

PROMPT = "Why is the sky blue?"

# Rough pretraining data diet — illustrative proportions, not SmolLM2's exact
# mix. Just enough to show "it reads a bit of everything".
DATA_MIX = [
    {"label": "Pagine web", "pct": 58},
    {"label": "Codice",     "pct": 18},
    {"label": "Libri",      "pct": 14},
    {"label": "Wikipedia",  "pct": 10},
]

# --- AUTHORED content (used in both modes; real mode overrides two answers) --
AUTHORED = {
    "schema_version": "1.0",
    "model": {"base": BASE_MODEL, "instruct": INSTRUCT_MODEL},
    "prompt": PROMPT,
    "data_mix": DATA_MIX,

    # ① PRETRAINING — the base model doesn't answer, it continues the pattern.
    # (Real mode replaces `answer` with the model's actual continuation.)
    "pretrain": {
        "answer": (
            "Why is the grass green? Why is the sea salty? Why do we dream? "
            "Write your answers in your notebook and bring them to class on "
            "Monday. 4. Why is the sky blue? 5. Why is snow white?"),
        "verbatim": False,
        "caption": (
            "Non ha mai imparato a *rispondere* — ha solo imparato a indovinare "
            "quale testo viene di solito dopo. Dopo una domanda come questa, "
            "internet è pieno di altre domande, quindi è proprio quello che "
            "scrive."),
    },

    # ② FINE-TUNING — shown thousands of (instruction → response) pairs, it
    # learns the SHAPE of helping: a question should be answered.
    "finetune": {
        "examples": [
            {"instruction": "Explain why leaves are green.",
             "response": "Leaves are green because they are full of a pigment "
                         "called chlorophyll, which reflects green light."},
            {"instruction": "Give me three names for a pet rabbit.",
             "response": "Sure! How about Clover, Pepper, or Biscuit?"},
            {"instruction": "Turn this into a polite sentence: gimme the salt.",
             "response": "Could you please pass me the salt?"},
        ],
        # Helpful and well-shaped — but it slips in a common myth (the ocean
        # does NOT make the sky blue). RL fixes this in the next stage.
        "answer": (
            "Great question! The sky looks blue partly because the ocean "
            "reflects up onto it, and partly because the air scatters blue "
            "light during the day. That's why we see a nice blue color above "
            "us. 🌤️"),
        "verbatim": False,
        "caption": (
            "Ora risponde con un formato chiaro e cordiale — un salto enorme. "
            "Ma imitare un testo che suona utile non è lo stesso che avere "
            "ragione: questa risposta ripete un mito molto diffuso."),
    },

    # ③ RL — graded answers. Two flavours of reward, both kid-legible.
    "rl": {
        # (a) Human-preference reward: the learner BECOMES the reward model.
        "preference": {
            "question": PROMPT,
            "candidates": [
                {"better": False, "tag": "suona bene, ma è sbagliata",
                 "text": "The sky is blue because the ocean reflects up onto "
                         "it and colors the air above us."},
                {"better": True, "tag": "chiara e corretta",
                 "text": "Sunlight is a mix of all colors. As it travels "
                         "through the air, the tiny gas molecules scatter the "
                         "short, blue waves much more than the others, so blue "
                         "light spreads across the whole sky. (The blue ocean "
                         "is a separate thing — the sky looks blue even over a "
                         "desert.)"},
            ],
            "reward_up": "Il modello viene spinto a dare più risposte come "
                         "questa.",
            "reward_down": "Il modello viene spinto ad allontanarsi da "
                           "risposte come questa.",
            "reveal": "Chi addestra il modello premia le risposte chiare E "
                      "corrette. La seconda spiega il vero motivo e corregge "
                      "anche il mito dell'oceano — quindi è da questa che il "
                      "modello impara.",
        },
        # (b) Verifiable reward: no opinions, just "is the number right?"
        "verifiable": {
            "problem": "A notebook costs $3. Pens cost $2 each. How much for "
                       "1 notebook and 4 pens?",
            "naive_answer": "That's $5.",
            "naive_note": "Ha riconosciuto lo schema “somma i due numeri” "
                          "(3 + 2) e ha sparato una risposta.",
            "thinking": [
                "1 notebook = $3",
                "4 pens = 4 × $2 = $8",
                "$3 + $8 = $11",
            ],
            "answer": "$11",
            "reward_note": (
                "Qui la ricompensa è semplicissima: il numero finale è "
                "giusto? Non serve nessuna opinione. Allenarsi su questo voto "
                "sì/no è esattamente ciò che spinge il modello a rallentare "
                "e mostrare il proprio ragionamento — quello che si chiama "
                "il suo “pensiero”."),
        },
    },

    # The three report cards shown in the outro — the same question, grown up.
    # NOTE: "kind" is a structural key (matches stages.parrot/helper/expert in
    # training.js, used as a CSS class + lookup key) — keep it in English.
    "report_cards": [
        {"stage": "Pre-addestrato", "kind": "parrot",
         "verdict": "Completamento automatico. Continua il tuo testo invece "
                    "di rispondere."},
        {"stage": "Fine-tuned", "kind": "helper",
         "verdict": "Risponde in modo utile — ma può ripetere errori con "
                    "sicurezza."},
        {"stage": "Dopo l'RL", "kind": "expert",
         "verdict": "Più preciso, e mostra il suo ragionamento."},
    ],
}


# ---------------------------------------------------------------------------
# REAL: SmolLM2 base (continues) vs SmolLM2-Instruct (answers)
# ---------------------------------------------------------------------------
def _trim(text, max_chars=320):
    """Tidy a tiny model's raw output: collapse whitespace, keep whole
    sentences, cap length so a classroom card stays readable."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # back off to the last sentence end so we don't stop mid-word
    m = re.search(r"[.!?][^.!?]*$", cut)
    return (cut[: m.start() + 1] if m else cut).strip()


# Captions for REAL mode. The AUTHORED captions above describe the authored
# stand-in answers; these describe what SmolLM2 actually writes under greedy
# decoding (deterministic — re-check them if the models or the generate()
# parameters ever change).
REAL_PRETRAIN_CAPTION = (
    "Non ha mai imparato a rispondere — solo a far continuare il testo. "
    "Quindi, invece di dirti perché, scrive quello che direbbe dopo un "
    "articolo su questa domanda: un grande incipit, e nessuna risposta in "
    "vista.")
REAL_FINETUNE_CAPTION = (
    "Ora risponde con un formato chiaro e sicuro — un salto enorme. Ma "
    "leggi bene: attribuisce il fenomeno a particelle di “aerosol” che "
    "galleggiano nell'aria (in realtà sono le stesse molecole di gas "
    "dell'aria), e dice che il viola ha una lunghezza d'onda più lunga, "
    "quando è una delle più corte. Essere fluenti non è lo stesso che avere "
    "ragione.")


def build_real():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    doc = json.loads(json.dumps(AUTHORED))  # deep copy

    # --- base model: feed the raw prompt, watch it CONTINUE (not answer) ---
    print(f"loading base {BASE_MODEL} …")
    btok = AutoTokenizer.from_pretrained(BASE_MODEL)
    bmodel = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    bmodel.eval()
    bin_ = btok(PROMPT, return_tensors="pt")
    with torch.no_grad():
        bout = bmodel.generate(**bin_, max_new_tokens=60, do_sample=False,
                               repetition_penalty=1.3,
                               pad_token_id=btok.eos_token_id)
    cont = _trim(btok.decode(bout[0][bin_["input_ids"].shape[1]:],
                             skip_special_tokens=True))
    if cont:
        doc["pretrain"]["answer"] = cont
        doc["pretrain"]["verbatim"] = True
        doc["pretrain"]["caption"] = REAL_PRETRAIN_CAPTION
    print(f"  base continuation: {doc['pretrain']['answer'][:80]}…")

    # --- instruct model: same question through the chat template -> it ANSWERS ---
    print(f"loading instruct {INSTRUCT_MODEL} …")
    itok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL)
    imodel = AutoModelForCausalLM.from_pretrained(INSTRUCT_MODEL, torch_dtype=torch.float32)
    imodel.eval()
    msgs = [{"role": "user", "content": PROMPT}]
    iin = itok.apply_chat_template(msgs, add_generation_prompt=True,
                                   return_tensors="pt")
    with torch.no_grad():
        iout = imodel.generate(iin, max_new_tokens=120, do_sample=False,
                               repetition_penalty=1.2,
                               pad_token_id=itok.eos_token_id)
    ans = _trim(itok.decode(iout[0][iin.shape[1]:], skip_special_tokens=True))
    if ans:
        doc["finetune"]["answer"] = ans
        doc["finetune"]["verbatim"] = True
        doc["finetune"]["caption"] = REAL_FINETUNE_CAPTION
    print(f"  instruct answer: {doc['finetune']['answer'][:80]}…")
    print("  (RL stage stays hand-authored — see module docstring)")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true",
                    help="write hand-authored data (no model download)")
    args = ap.parse_args()

    doc = AUTHORED if args.fixture else build_real()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w"), ensure_ascii=False, separators=(",", ":"))
    mode = "AUTHORED" if args.fixture else "REAL base+instruct"
    print(f"wrote {os.path.relpath(OUT)}  ({mode})")


if __name__ == "__main__":
    main()
