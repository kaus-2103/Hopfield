"""
Fuzzy Hopfield Network - Mini Project
=====================================
Noisy pattern recall (7x7 letters) with a Hopfield associative memory
whose neurons use FUZZY membership functions instead of a hard sign().

Everything is written from scratch. The only library used is NumPy, for
array maths. No scikit-learn / TensorFlow / PyTorch / ready-made models.

Pipeline
--------
1. Raw data      : 7x7 letters rendered as grayscale intensities (0-255) + noise
2. Pre-processing: clip -> min-max normalise -> fuzzify (S-function) -> bipolar
3. Training      : Hebbian rule (one-shot, no iterations)
4. Recall        : asynchronous updates, neuron = fuzzy sigmoid membership,
                   steepness annealed (soft -> sharp)
5. Fuzzy decision: fuzzy Jaccard similarity + fuzzy rules -> confidence/reject
6. Experiment    : classic (hard sign) Hopfield vs fuzzy Hopfield vs noise level
"""

import numpy as np

# ----------------------------------------------------------------------
# 0. Stored patterns (7x7, '#' = ON pixel)
# ----------------------------------------------------------------------
LETTERS = {
    "A": ["..###..",
          ".#...#.",
          "#.....#",
          "#######",
          "#.....#",
          "#.....#",
          "#.....#"],
    "C": [".#####.",
          "#.....#",
          "#......",
          "#......",
          "#......",
          "#.....#",
          ".#####."],
    "H": ["#.....#",
          "#.....#",
          "#.....#",
          "#######",
          "#.....#",
          "#.....#",
          "#.....#"],
    "L": ["#......",
          "#......",
          "#......",
          "#......",
          "#......",
          "#......",
          "#######"],
    "T": ["#######",
          "...#...",
          "...#...",
          "...#...",
          "...#...",
          "...#...",
          "...#..."],
}
SHAPE = (7, 7)


# ----------------------------------------------------------------------
# 1. RAW DATA GENERATION (simulates a noisy sensor / scanner)
# ----------------------------------------------------------------------
def ascii_to_clean_image(rows):
    """'#'/'.' text -> 7x7 array of 0 or 255 intensities."""
    return np.array([[255.0 if ch == "#" else 0.0 for ch in r] for r in rows])


def make_noisy_raw(clean_img, rng, gauss_sigma=40.0, flip_prob=0.1):
    """
    Corrupt a clean image the way a real sensor would:
      * Gaussian intensity noise (sensor noise)
      * salt & pepper flips (dead / hot pixels)
    Result is deliberately allowed to go outside 0..255 (needs clipping).
    """
    img = clean_img + rng.normal(0.0, gauss_sigma, clean_img.shape)
    flips = rng.random(clean_img.shape) < flip_prob
    img[flips] = 255.0 - clean_img[flips]
    return img


# ----------------------------------------------------------------------
# 2. PRE-PROCESSING
# ----------------------------------------------------------------------
def clip_range(img, lo=0.0, hi=255.0):
    return np.clip(img, lo, hi)


def min_max_normalize(img):
    """Scale to [0, 1]. Guard against a flat (constant) image."""
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-9:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def s_membership(x, a=0.3, c=0.7):
    """
    Fuzzy S-function membership for the fuzzy set "pixel is ON".
        x <= a        -> 0
        a < x <= mid  -> 2*((x-a)/(c-a))^2
        mid < x < c   -> 1 - 2*((x-c)/(c-a))^2
        x >= c        -> 1
    A pixel at 0.5 intensity is therefore 'half ON' instead of forced to 0 or 1.
    """
    x = np.asarray(x, dtype=float)
    mid = (a + c) / 2.0
    mu = np.zeros_like(x)
    mu[x >= c] = 1.0
    m1 = (x > a) & (x <= mid)
    m2 = (x > mid) & (x < c)
    mu[m1] = 2.0 * ((x[m1] - a) / (c - a)) ** 2
    mu[m2] = 1.0 - 2.0 * ((x[m2] - c) / (c - a)) ** 2
    return mu


def to_bipolar(mu):
    """Membership [0,1] -> bipolar [-1,1] (0.5 becomes 0 = 'undecided')."""
    return 2.0 * mu - 1.0


def preprocess(raw_img):
    """Full pipeline. Returns (fuzzy membership vector, bipolar vector)."""
    x = clip_range(raw_img)
    x = min_max_normalize(x)
    mu = s_membership(x).ravel()          # flatten 7x7 -> 49
    return mu, to_bipolar(mu)


# ----------------------------------------------------------------------
# 3. HOPFIELD NETWORK (from scratch)
# ----------------------------------------------------------------------
class FuzzyHopfield:
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))

    # --- training: Hebbian outer-product rule --------------------------
    def train(self, patterns):
        """patterns: array (P, N) of bipolar vectors (+1 / -1)."""
        P = len(patterns)
        W = np.zeros((self.n, self.n))
        for p in patterns:
            W += np.outer(p, p)
        W /= self.n
        np.fill_diagonal(W, 0.0)          # no self connections
        self.W = W

    # --- fuzzy neuron --------------------------------------------------
    @staticmethod
    def fuzzy_activation(net, k):
        """
        Membership of 'neuron is ON' = sigmoid(k * net)  in [0,1]
        mapped back to bipolar: 2*mu - 1 (= tanh(k*net/2)).
        Small k  -> very soft / fuzzy (uncertain neurons stay uncertain)
        Large k  -> approaches hard sign().
        """
        mu = 1.0 / (1.0 + np.exp(-np.clip(k * net, -50, 50)))
        return 2.0 * mu - 1.0

    def energy(self, s):
        return -0.5 * s @ self.W @ s

    # --- recall --------------------------------------------------------
    def recall_fuzzy(self, s0, rng, k_start=4.0, k_end=16.0,
                     max_sweeps=30, tol=1e-4, beta=0.5):
        """
        Asynchronous update with annealed fuzzy activation.
        beta = weight of the external evidence: the pre-processed fuzzy input
        s0 keeps pulling every neuron toward what the sensor actually saw
        (a confident pixel pulls hard, an uncertain ~0 pixel barely pulls).
        Without it, soft states can drift into the wrong attractor.
        """
        s = s0.copy()
        history = [self.energy(s)]
        for sweep in range(max_sweeps):
            k = k_start + (k_end - k_start) * sweep / max(1, max_sweeps - 1)
            s_old = s.copy()
            for i in rng.permutation(self.n):
                s[i] = self.fuzzy_activation(self.W[i] @ s + beta * s0[i], k)
            history.append(self.energy(s))
            if np.max(np.abs(s - s_old)) < tol:
                break
        return s, history

    def recall_classic(self, s0, rng, max_sweeps=30):
        """Standard Hopfield: hard thresholding, input forced to +-1."""
        s = np.where(s0 >= 0, 1.0, -1.0)
        for _ in range(max_sweeps):
            changed = False
            for i in rng.permutation(self.n):
                new = 1.0 if self.W[i] @ s >= 0 else -1.0
                if new != s[i]:
                    s[i] = new
                    changed = True
            if not changed:
                break
        return s


# ----------------------------------------------------------------------
# 4. FUZZY DECISION LAYER (fuzzy similarity + fuzzy rules)
# ----------------------------------------------------------------------
def fuzzy_jaccard(mu_a, mu_b):
    """Fuzzy Jaccard similarity = sum(min) / sum(max)  (min = AND, max = OR)."""
    return np.minimum(mu_a, mu_b).sum() / (np.maximum(mu_a, mu_b).sum() + 1e-12)


def trapezoid(x, a, b, c, d):
    """Trapezoidal membership function (a<=b<=c<=d)."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a + 1e-12)
    return (d - x) / (d - c + 1e-12)


def fuzzy_decide(sims):
    """
    sims : dict {label: similarity of recalled state to that stored pattern}
    Fuzzy rules:
        R1: IF best_sim is HIGH   AND  margin is LARGE  THEN confidence HIGH
        R2: IF best_sim is MEDIUM OR   margin is SMALL  THEN confidence LOW
    Confidence = defuzzified with a simple weighted average (1.0 / 0.2).
    Returns (label, confidence in [0,1], verdict string)
    """
    ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best - second

    best_high = trapezoid(best, 0.55, 0.85, 1.01, 1.02)
    best_med = trapezoid(best, 0.20, 0.40, 0.55, 0.85)
    margin_large = trapezoid(margin, 0.10, 0.30, 1.01, 1.02)
    margin_small = trapezoid(margin, -0.01, 0.0, 0.08, 0.25)

    r1 = min(best_high, margin_large)       # AND = min
    r2 = max(best_med, margin_small)        # OR  = max
    conf = (r1 * 1.0 + r2 * 0.2) / (r1 + r2 + 1e-12)

    if conf >= 0.75:
        verdict = "ACCEPT"
    elif conf >= 0.45:
        verdict = "UNSURE"
    else:
        verdict = "REJECT"
    return best_label, conf, verdict


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def show(vec, title=""):
    """Print a vector (bipolar or membership) as ASCII art."""
    v = np.asarray(vec).reshape(SHAPE)
    if v.min() < 0:                       # bipolar -> membership
        v = (v + 1) / 2
    glyph = lambda m: "#" if m > 0.75 else ("+" if m > 0.5 else ("." if m > 0.25 else " "))
    print(title)
    for row in v:
        print("   " + "".join(glyph(m) * 2 for m in row))


def to_label(s, labels, patterns):
    """Nearest stored pattern by dot-product overlap."""
    return labels[int(np.argmax(patterns @ s))]


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)

    # ---- build stored memories (through the same preprocessing) --------
    labels = list(LETTERS.keys())
    clean_imgs = {k: ascii_to_clean_image(v) for k, v in LETTERS.items()}
    mu_store = np.array([preprocess(clean_imgs[k])[0] for k in labels])
    bip_store = np.array([preprocess(clean_imgs[k])[1] for k in labels])

    net = FuzzyHopfield(n_neurons=SHAPE[0] * SHAPE[1])
    net.train(bip_store)
    print(f"Trained on {len(labels)} patterns: {labels}  "
          f"({net.n} neurons, {net.n ** 2 - net.n} weights)\n")

    # ---- single demo ----------------------------------------------------
    target = "H"
    raw = make_noisy_raw(clean_imgs[target], rng, gauss_sigma=50, flip_prob=0.15)
    mu_in, bip_in = preprocess(raw)
    show(clean_imgs[target] / 255.0, f"Original '{target}'")
    show(mu_in, "Noisy input after pre-processing (fuzzy memberships)")

    out_fuzzy, hist = net.recall_fuzzy(bip_in, rng)
    show(out_fuzzy, "Fuzzy Hopfield output")
    mu_out = (out_fuzzy + 1) / 2
    sims = {l: fuzzy_jaccard(mu_out, mu_store[i]) for i, l in enumerate(labels)}
    lab, conf, verdict = fuzzy_decide(sims)
    print(f"\nPredicted: {lab} | confidence {conf:.2f} | {verdict}")
    print("Similarities:", {k: round(v, 2) for k, v in sims.items()})
    print(f"Energy: start {hist[0]:.2f} -> end {hist[-1]:.2f} "
          f"({len(hist) - 1} sweeps)\n")

    # ---- experiment: accuracy vs noise ---------------------------------
    print("Accuracy vs noise (200 trials per pattern per level)")
    print(f"{'flip%':>6} {'gauss':>6} | {'classic':>8} {'fuzzy':>8} | "
          f"{'fuzzy ACCEPT-acc':>17}")
    trials = 200
    for flip in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        sigma = 30 + 100 * flip           # noise grows together
        ok_c = ok_f = n = acc_n = acc_ok = 0
        for li, l in enumerate(labels):
            for _ in range(trials):
                raw = make_noisy_raw(clean_imgs[l], rng, sigma, flip)
                mu_in, bip_in = preprocess(raw)

                s_c = net.recall_classic(bip_in, rng)
                ok_c += to_label(s_c, labels, bip_store) == l

                s_f, _ = net.recall_fuzzy(bip_in, rng)
                mu_o = (s_f + 1) / 2
                sims = {x: fuzzy_jaccard(mu_o, mu_store[j])
                        for j, x in enumerate(labels)}
                pred, conf, verdict = fuzzy_decide(sims)
                ok_f += pred == l
                if verdict == "ACCEPT":
                    acc_n += 1
                    acc_ok += pred == l
                n += 1
        acc_txt = f"{100 * acc_ok / acc_n:6.1f}%" if acc_n else "   n/a"
        print(f"{flip * 100:6.0f} {sigma:6.0f} | {100 * ok_c / n:7.1f}% "
              f"{100 * ok_f / n:7.1f}% | {acc_txt:>17}")


if __name__ == "__main__":
    main()
