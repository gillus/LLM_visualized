"""Offline generator for the "Rete neurale" tab (web/neural-net.html).

Dumps web/house_prices.json: a small SYNTHETIC dataset of houses for an
imaginary estate agency — the running price-predictor example from the lesson
slides. Unlike the other tabs there is no language model involved: the browser
trains tiny models on this data live (web/py/nnlab.py, run by PyScript).

    python3 generate/house_prices.py

Stdlib only and seeded, so re-running reproduces the identical file.

The hidden "true" price rule is deliberately NOT a straight line in every
direction: being close to the centre adds a premium that switches on fairly
sharply (an S-curve in distance). That is the one thing a single line — and a
single neuron — can't capture, and two neurons can: the tab's whole story.
"""
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "web", "house_prices.json")

SEED = 7
N_TRAIN = 80
N_TEST = 40

# Order matters: feature 0 is the one the single-line regression uses.
FEATURES = [
    {"key": "superficie", "label": "superficie", "unit": "m²", "decimals": 0},
    {"key": "distanza", "label": "distanza dal centro", "unit": "km", "decimals": 1},
    {"key": "anno", "label": "anno di costruzione", "unit": "", "decimals": 0},
]

# The hidden rule (€). Documented in SCHEMA.md so teachers can reveal it.
BASE = 20_000
PER_M2 = 1_600
CENTRE_PREMIUM = 130_000     # full premium right in the centre…
CENTRE_EDGE_KM = 2.5         # …halved at this distance…
CENTRE_SOFTNESS_KM = 0.45    # …and gone within ~1 km past it
PER_YEAR = 900               # newer houses are worth a bit more
NOISE_SD = 14_000            # everything the agency can't see


def centre_bonus(km):
    return CENTRE_PREMIUM / (1 + math.exp((km - CENTRE_EDGE_KM) / CENTRE_SOFTNESS_KM))


def make_house(rng):
    # apartments (small) are more common than villas (large)
    m2 = round(40 + 160 * rng.random() ** 1.4)
    km = round(0.2 + 7.8 * rng.random() ** 1.3, 1)
    year = rng.randint(1950, 2022)
    price = (BASE + PER_M2 * m2 + centre_bonus(km) + PER_YEAR * (year - 1950)
             + rng.gauss(0, NOISE_SD))
    return {"x": [m2, km, year], "price": int(round(price, -3))}


def main():
    rng = random.Random(SEED)
    houses = [make_house(rng) for _ in range(N_TRAIN + N_TEST)]
    data = {
        "schema_version": "1.0",
        "synthetic": True,
        "features": FEATURES,
        "train": houses[:N_TRAIN],
        "test": houses[N_TRAIN:],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {os.path.relpath(OUT)}: {N_TRAIN} train + {N_TEST} test houses")


if __name__ == "__main__":
    main()
