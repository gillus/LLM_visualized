# nnlab.py — the "brain" of the Rete neurale tab.
#
# Everything here is plain Python with no libraries beyond math/random, so the
# same file runs in the browser (PyScript + MicroPython) and on a laptop
# (python3). The page's JavaScript only DRAWS: every number you see — the
# line, the neurons, the errors — is computed by this file.
#
# Two models learn to guess a house's price:
#   * the LINE:  price = slope × m² + base          (2 knobs)
#   * a small NEURAL NETWORK: each neuron weighs its inputs, adds them up and
#     squashes the total into 0…1 with an S-curve (a "logistic regression");
#     the output neuron adds the last layer's answers into a price.
# Training = "try, measure, correct", repeated: measure how wrong the guesses
# are on the training houses, then nudge every knob a little in the direction
# that makes the error smaller (gradient descent).

import math
import random

# hidden-layer sizes for each architecture the page offers
ARCHS = {"1": [1], "2": [2], "2x2": [2, 2]}

CURVE_POINTS = 48   # samples along a direction for the drawn curves


def sigmoid(z):
    if z > 30:
        z = 30.0
    elif z < -30:
        z = -30.0
    return 1.0 / (1.0 + math.exp(-z))


def gauss():
    # Box–Muller: a bell-curve random number (MicroPython has no random.gauss)
    u = random.random() or 1e-12
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * random.random())


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) or 1.0


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


class Lab:
    def __init__(self, data):
        self.features = data["features"]
        self.nf = len(self.features)
        train, test = data["train"], data["test"]
        cols = [[h["x"][j] for h in train] for j in range(self.nf)]
        # Inputs and prices are rescaled to "about −2…+2" before training —
        # same knobs, but gradient descent behaves far better on small numbers.
        self.mu = [mean(c) for c in cols]
        self.sd = [std(c) for c in cols]
        prices = [h["price"] for h in train]
        self.ymu, self.ysd = mean(prices), std(prices)
        self.Xtr = [self.norm_x(h["x"]) for h in train]
        self.ytr = [self.norm_y(h["price"]) for h in train]
        self.Xte = [self.norm_x(h["x"]) for h in test]
        self.yte_eur = [h["price"] for h in test]
        self.ytr_eur = prices
        allh = train + test
        self.lo = [min(h["x"][j] for h in allh) for j in range(self.nf)]
        self.hi = [max(h["x"][j] for h in allh) for j in range(self.nf)]
        self.typical = [median(c) for c in cols]   # the "casa tipica"
        self.line_reset()
        self.nn_reset("2")

    # ---------- scaling ----------
    def norm_x(self, x):
        return [(x[j] - self.mu[j]) / self.sd[j] for j in range(self.nf)]

    def norm_y(self, p):
        return (p - self.ymu) / self.ysd

    def eur(self, yn):
        return yn * self.ysd + self.ymu

    def mae_eur(self, preds_n, truth_eur):
        return mean([abs(self.eur(p) - t) for p, t in zip(preds_n, truth_eur)])

    # =================================================================
    # THE LINE  (feature 0 only: superficie)
    # =================================================================
    def line_reset(self):
        # "parte con numeri a caso": almost flat and far too low
        self.la = 0.05 + 0.1 * random.random()
        self.lc = -0.9 - 0.4 * random.random()
        self.lva = self.lvc = 0.0
        self.line_steps = 0

    def line_set(self, slope_eur, base_eur):
        # the learner moved the sliders: convert € knobs → scaled knobs
        s0 = self.sd[0]
        self.la = slope_eur * s0 / self.ysd
        self.lc = (base_eur + slope_eur * self.mu[0] - self.ymu) / self.ysd
        self.lva = self.lvc = 0.0

    def line_step(self, n=1, lr=0.06, momentum=0.85):
        X, Y = self.Xtr, self.ytr
        m = len(Y)
        for _ in range(n):
            ga = gc = 0.0
            for x, y in zip(X, Y):
                err = self.la * x[0] + self.lc - y        # measure
                ga += err * x[0]
                gc += err
            # correct (with a little momentum, so it overshoots and settles)
            self.lva = momentum * self.lva - lr * 2 * ga / m
            self.lvc = momentum * self.lvc - lr * 2 * gc / m
            self.la += self.lva
            self.lc += self.lvc
            self.line_steps += 1

    def line_state(self):
        slope = self.la * self.ysd / self.sd[0]
        base = self.ymu + self.lc * self.ysd - slope * self.mu[0]
        ptr = [self.la * x[0] + self.lc for x in self.Xtr]
        pte = [self.la * x[0] + self.lc for x in self.Xte]
        return {
            "slope": slope, "base": base, "steps": self.line_steps,
            "mae_train": self.mae_eur(ptr, self.ytr_eur),
            "mae_test": self.mae_eur(pte, self.yte_eur),
            "params": 2,
        }

    # =================================================================
    # THE NEURAL NETWORK
    # =================================================================
    def nn_reset(self, arch, seed=None):
        if seed is not None:
            random.seed(seed)
        self.arch = arch
        sizes = [self.nf] + ARCHS[arch] + [1]
        self.layers = []
        for li in range(len(sizes) - 1):
            n_in, n_out = sizes[li], sizes[li + 1]
            scale = 1.0 / math.sqrt(n_in)
            W = [[gauss() * scale for _ in range(n_in)] for _ in range(n_out)]
            b = [0.0] * n_out
            self.layers.append({"W": W, "b": b})
        # Adam's running averages, one per knob
        self.adam = [{"mW": [[0.0] * len(r) for r in L["W"]],
                      "vW": [[0.0] * len(r) for r in L["W"]],
                      "mb": [0.0] * len(L["b"]), "vb": [0.0] * len(L["b"])}
                     for L in self.layers]
        self.epochs = 0

    def n_params(self):
        return sum(len(L["b"]) + sum(len(r) for r in L["W"]) for L in self.layers)

    def forward(self, x):
        """Return the list of every layer's outputs: [inputs, hidden…, price]."""
        acts = [x]
        last = len(self.layers) - 1
        for li, L in enumerate(self.layers):
            prev = acts[-1]
            out = []
            for w_row, b in zip(L["W"], L["b"]):
                z = b
                for w, a in zip(w_row, prev):
                    z += w * a
                out.append(z if li == last else sigmoid(z))
            acts.append(out)
        return acts

    def nn_step(self, n=1, lr=0.06, b1=0.9, b2=0.999, eps=1e-8):
        layers = self.layers
        nl = len(layers)
        m = len(self.ytr)
        for _ in range(n):
            gW = [[[0.0] * len(r) for r in L["W"]] for L in layers]
            gb = [[0.0] * len(L["b"]) for L in layers]
            for x, y in zip(self.Xtr, self.ytr):
                acts = self.forward(x)
                # error at the output (gradient of the squared error)
                delta = [2.0 * (acts[-1][0] - y) / m]
                for li in range(nl - 1, -1, -1):
                    L = layers[li]
                    a_in = acts[li]
                    for o, d in enumerate(delta):
                        gb[li][o] += d
                        row = gW[li][o]
                        for i, a in enumerate(a_in):
                            row[i] += d * a
                    if li:
                        # pass the blame back through the S-curves
                        back = []
                        for i, a in enumerate(a_in):
                            s = 0.0
                            for o, d in enumerate(delta):
                                s += L["W"][o][i] * d
                            back.append(s * a * (1.0 - a))
                        delta = back
            # Adam: a smarter "nudge" that adapts the step size per knob
            self.epochs += 1
            t = self.epochs
            c1 = 1.0 - b1 ** t
            c2 = 1.0 - b2 ** t
            for li, L in enumerate(layers):
                A = self.adam[li]
                for o, w_row in enumerate(L["W"]):
                    mr, vr, gr = A["mW"][o], A["vW"][o], gW[li][o]
                    for i in range(len(w_row)):
                        g = gr[i]
                        mr[i] = b1 * mr[i] + (1 - b1) * g
                        vr[i] = b2 * vr[i] + (1 - b2) * g * g
                        w_row[i] -= lr * (mr[i] / c1) / (math.sqrt(vr[i] / c2) + eps)
                    g = gb[li][o]
                    A["mb"][o] = b1 * A["mb"][o] + (1 - b1) * g
                    A["vb"][o] = b2 * A["vb"][o] + (1 - b2) * g * g
                    L["b"][o] -= lr * (A["mb"][o] / c1) / (math.sqrt(A["vb"][o] / c2) + eps)

    def predict_n(self, xn):
        return self.forward(xn)[-1][0]

    def nn_state(self, direction=0):
        """Everything the page draws, for one 'direction' (input feature)."""
        d = direction
        xs, curve = [], []
        hidden = [[[] for _ in L["b"]] for L in self.layers[:-1]]
        for k in range(CURVE_POINTS):
            v = self.lo[d] + (self.hi[d] - self.lo[d]) * k / (CURVE_POINTS - 1)
            house = list(self.typical)
            house[d] = v
            acts = self.forward(self.norm_x(house))
            xs.append(v)
            curve.append(self.eur(acts[-1][0]))
            for li in range(len(hidden)):
                for ni in range(len(hidden[li])):
                    hidden[li][ni].append(acts[li + 1][ni])
        pte = [self.predict_n(x) for x in self.Xte]
        ptr = [self.predict_n(x) for x in self.Xtr]
        return {
            "arch": self.arch, "epochs": self.epochs, "params": self.n_params(),
            "layers": [{"W": L["W"], "b": L["b"]} for L in self.layers],
            "direction": d, "xs": xs, "curve": curve, "hidden": hidden,
            "typical": self.typical,
            "test_pred": [self.eur(p) for p in pte],
            "mae_train": self.mae_eur(ptr, self.ytr_eur),
            "mae_test": self.mae_eur(pte, self.yte_eur),
        }
