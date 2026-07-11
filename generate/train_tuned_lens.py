"""Train a tuned lens for SmolLM2 and cache it to generate/tuned_lens.pt.
Then re-run generate.py — it will automatically emit a `logit_lens_tuned`
field and the website shows a Plain/Tuned toggle.

    python generate/train_tuned_lens.py                 # defaults (wikitext-2)
    python generate/train_tuned_lens.py --steps 400
    python generate/train_tuned_lens.py --text mytext.txt

Each layer's translator is trained to match the model's FINAL next-token
distribution (KL divergence), so its readout reflects what the model actually
ends up predicting — denoising the middle layers the plain lens shows as junk.

GPU strongly recommended (the per-layer unembedding over a 49k vocab is the
cost). On the model author's machine this is a few minutes; CPU is much slower.
"""
import argparse
import os

import torch
import torch.nn.functional as F

from tuned_lens_model import TunedLens, save_tuned_lens

MODEL = "HuggingFaceTB/SmolLM2-135M"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tuned_lens.pt")


def get_token_blocks(tok, seqlen, max_blocks, text_path=None):
    if text_path:
        text = open(text_path, encoding="utf-8").read()
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    n = (len(ids) // seqlen) * seqlen
    blocks = torch.tensor(ids[:n]).view(-1, seqlen)
    if blocks.shape[0] > max_blocks:
        blocks = blocks[:max_blocks]
    print(f"corpus: {len(ids):,} tokens -> {blocks.shape[0]} blocks of {seqlen}")
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-blocks", type=int, default=400)
    ap.add_argument("--text", type=str, default=None, help="local .txt corpus instead of wikitext")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    norm, lm_head = model.model.norm, model.lm_head
    L = model.config.num_hidden_layers          # translators for hidden_states[0..L-1]
    d = model.config.hidden_size
    lens = TunedLens(L, d).to(device)
    opt = torch.optim.Adam(lens.parameters(), lr=args.lr)

    blocks = get_token_blocks(tok, args.seqlen, args.max_blocks, args.text).to(device)
    nb = blocks.shape[0]

    for step in range(args.steps):
        idx = torch.randint(0, nb, (args.batch,), device=device)
        ids = blocks[idx]
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
            target = torch.softmax(out.logits, dim=-1)          # teacher = final dist
            hs = [h.detach() for h in out.hidden_states]
        V = target.shape[-1]
        opt.zero_grad()
        running = 0.0
        for li in range(L):                                     # backward per layer = low memory
            pred = lm_head(norm(lens(li, hs[li])))              # (B,T,V)
            logp = F.log_softmax(pred, dim=-1)
            loss_li = F.kl_div(logp.view(-1, V), target.view(-1, V),
                               reduction="batchmean") / L
            loss_li.backward()
            running += loss_li.item()
        opt.step()
        if step % 25 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}/{args.steps}  KL/layer {running:.4f}")

    save_tuned_lens(lens, MODEL, OUT)
    print(f"saved tuned lens -> {OUT}")
    print("now re-run: python generate/generate.py")


if __name__ == "__main__":
    main()
