# Fuzzy Hopfield Network: Mini Project

A Hopfield associative memory that uses **fuzzy logic** at three levels (input, neuron activation and output decision) to recover noisy 7×7 letter images.

Everything is implemented from scratch. **NumPy is the only dependency**, used purely for array maths. There is no scikit-learn, TensorFlow, PyTorch or any other ready-made model.

---

## Contents

1. [What the project does](#what-the-project-does)
2. [Quick start](#quick-start)
3. [Pipeline overview](#pipeline-overview)
4. [Step-by-step explanation](#step-by-step-explanation)
5. [Sample output and results](#sample-output-and-results)
6. [Design decisions](#design-decisions)
7. [Limitations](#limitations)
8. [Customising the project](#customising-the-project)
9. [Ideas for extension](#ideas-for-extension)

---

## What the project does

The network stores five letters (**A, C, H, L, T**) as 7×7 pixel patterns (49 neurons). It is then shown corrupted versions of those letters and has to recall the original.

It compares two recall methods:

| Method | Neuron behaviour | Input handling |
|---|---|---|
| **Classic Hopfield** | Hard threshold (`sign`) | Pixels forced to +1 / -1 |
| **Fuzzy Hopfield** | Smooth fuzzy membership, annealed | Pixels keep their degree of "ON-ness" |

The fuzzy version also produces a **confidence score** and an **ACCEPT / UNSURE / REJECT** verdict through a small fuzzy rule base.

## Quick start

**Requirements:** Python 3.8+ and NumPy.

```bash
pip install numpy
python fuzzy_hopfield.py
```

The script prints:

1. The number of patterns, neurons and weights.
2. An ASCII demo: original letter, noisy input, recalled output.
3. The predicted label, confidence, verdict, similarities and energy trace.
4. An accuracy table comparing classic and fuzzy Hopfield across noise levels.

The random seed is fixed (`42`) so a given setup is reproducible.

## Pipeline overview

```
 clean letter (ASCII)
        │
        ▼
 1. RAW DATA         grayscale 0-255 + Gaussian noise + pixel flips
        │
        ▼
 2. PRE-PROCESSING   clip → min-max normalise → fuzzify (S-function) → bipolar
        │
        ▼
 3. TRAINING         Hebbian outer-product rule (one shot)
        │
        ▼
 4. FUZZY RECALL     asynchronous updates, fuzzy sigmoid neuron,
                     annealed steepness, external evidence term
        │
        ▼
 5. FUZZY DECISION   fuzzy Jaccard similarity + fuzzy rules → confidence
        │
        ▼
   label + confidence + ACCEPT / UNSURE / REJECT
```

## Step-by-step explanation

### 1. Raw data generation

Each letter is defined as seven strings of `#` and `.`, where `#` is an ON pixel. `ascii_to_clean_image()` turns it into an array of intensities (255 for ON, 0 for OFF).

`make_noisy_raw()` then simulates a faulty sensor:

- **Gaussian noise** (`gauss_sigma`) models electrical or scanner noise.
- **Salt-and-pepper flips** (`flip_prob`) model dead or hot pixels. A flipped pixel becomes `255 - value`.

The result can go outside 0–255, so the raw data really needs cleaning.

### 2. Pre-processing

`preprocess()` applies four steps in order:

| Step | Function | Purpose |
|---|---|---|
| Clip | `clip_range` | Force values into 0–255 |
| Normalise | `min_max_normalize` | Scale to [0, 1]. A constant image is mapped to zeros to avoid division by zero |
| Fuzzify | `s_membership` | Convert intensity into a membership degree μ for the fuzzy set "pixel is ON" |
| Bipolar | `to_bipolar` | Map μ ∈ [0, 1] to [-1, 1] using `2μ − 1`, as Hopfield networks expect |

**The S-function membership** (default `a=0.3`, `c=0.7`):

```
x ≤ a                  →  μ = 0
a < x ≤ (a+c)/2        →  μ = 2·((x−a)/(c−a))²
(a+c)/2 < x < c        →  μ = 1 − 2·((x−c)/(c−a))²
x ≥ c                  →  μ = 1
```

A mid-grey pixel is therefore *partly ON* instead of being forced to a hard 0 or 1. After the bipolar mapping, a pixel with μ = 0.5 becomes 0, meaning "undecided".

### 3. Training (Hebbian rule)

For P stored bipolar patterns `p` and N neurons:

```
W = (1/N) · Σ  p · pᵀ ,   with W[i][i] = 0
```

Training is one shot, with no iterations or gradients. The memories are stored in the 49×49 weight matrix.

### 4. Fuzzy recall

The state starts as the pre-processed input. Neurons are then updated **asynchronously** in random order, sweep after sweep.

**Fuzzy neuron.** Instead of `sign(net)`, each neuron computes a membership degree of being ON:

```
μ = 1 / (1 + exp(−k · net))        output = 2μ − 1   (equals tanh(k·net / 2))
```

- Small `k` gives soft, fuzzy behaviour.
- Large `k` approaches a hard sign function.
- `k` is **annealed** linearly from `k_start = 4` to `k_end = 16` over the sweeps. The network starts soft and ends sharp.

**External evidence term.** The net input of each neuron is:

```
net_i = Σ_j W[i][j]·s_j  +  β · s0_i          (β = 0.5)
```

`s0` is the pre-processed fuzzy input. A pixel the sensor was sure about pulls strongly, and an uncertain pixel (near 0) barely pulls. See [Design decisions](#design-decisions) for why this is needed.

**Stopping.** Recall stops when no neuron changes by more than `1e-4` in a full sweep, or after 30 sweeps.

**Energy.** The script tracks `E = −½ sᵀWs`. It should fall as the network settles into a stored memory.

### 5. Fuzzy decision layer

After recall, the output is converted back to memberships and compared with each stored pattern using the **fuzzy Jaccard similarity**:

```
sim(A, B) = Σ min(Aᵢ, Bᵢ) / Σ max(Aᵢ, Bᵢ)       (min = fuzzy AND, max = fuzzy OR)
```

`fuzzy_decide()` takes the best similarity and the **margin** between the best and second-best, then applies two fuzzy rules using trapezoidal membership functions:

| Rule | Condition | Conclusion |
|---|---|---|
| R1 | best similarity is **HIGH** AND margin is **LARGE** | confidence is high |
| R2 | best similarity is **MEDIUM** OR margin is **SMALL** | confidence is low |

The rule strengths are combined by a weighted average (high = 1.0, low = 0.2), a simple defuzzification:

```
confidence = (r1·1.0 + r2·0.2) / (r1 + r2)
```

| Confidence | Verdict |
|---|---|
| ≥ 0.75 | **ACCEPT** |
| 0.45 – 0.75 | **UNSURE** |
| < 0.45 | **REJECT** |

### 6. Experiment

For each noise level, every letter is corrupted 200 times and recalled by both networks. Noise grows with the flip probability: `sigma = 30 + 100 × flip_prob`. Classic accuracy uses the nearest stored pattern by overlap. Fuzzy accuracy uses the highest fuzzy similarity. A final column reports the accuracy **only on the samples the fuzzy layer marked ACCEPT**.

## Sample output and results

Demo for the letter H with heavy noise (`sigma = 50`, 15% flips). `#` is strongly ON, `+` and `.` are partly ON:

```
Original 'H'                     Noisy input (fuzzy memberships)     Fuzzy Hopfield output
##          ##                               ##                                 ##
##          ##                     ##        ##                     ##..        ##
##          ##                   ##    ##    ##                     ##          ##
##############                   ####  ########                    ##############
##          ##                   ##        ##                      ##          ##
##          ##                   ##          ##                    ##          ##
##          ##                   ##      ######                    ##          ##
```

```
Predicted: H | confidence 1.00 | ACCEPT
```

Accuracy over 200 trials per pattern per noise level (results from one run):

| Pixel flips | Gaussian σ | Classic | Fuzzy | Fuzzy (ACCEPT only) |
|---|---|---|---|---|
| 0% | 30 | 100.0% | 100.0% | 100.0% |
| 5% | 35 | 97.3% | 99.9% | 100.0% |
| 10% | 40 | 93.5% | 99.3% | 99.6% |
| 15% | 45 | 87.5% | 97.9% | 99.4% |
| 20% | 50 | 83.1% | 93.9% | 98.0% |
| 25% | 55 | 74.9% | 85.5% | 94.4% |
| 30% | 60 | 63.5% | 73.1% | 86.6% |

**Reading the table:**

- The fuzzy network beats the classic one at every noise level above zero, and the gap is largest at moderate noise.
- The last column shows the confidence layer is meaningful: answers it accepts are more reliable than the average answer. This is the practical benefit of a fuzzy reject option.

Exact numbers can shift slightly with a different NumPy version or platform, but the trend should hold.

## Design decisions

**Why the evidence term (β) is needed.** My first fuzzy version used only `W·s` with a soft activation. It scored about 40% even on clean input. Soft, partly-ON states shrink the net input of neighbouring neurons, which lets the whole state drift into one dominant attractor (many letters collapsed into "H" or "L"). A hard sign avoids this because it saturates every neuron to ±1. Feeding the fuzzy input back into each neuron anchors the recall to what was actually observed, while keeping the fuzzy behaviour. Accuracy then rose to the figures above.

**Why anneal `k`?** Starting soft lets uncertain pixels be pulled by the stored memories instead of being committed too early. Ending sharp produces a clean final pattern.

**Why compare with a classic Hopfield?** The comparison isolates the effect of the fuzzy parts. Both networks share the same weights and the same noisy data.

**Tuned values.** The S-function bounds (0.3, 0.7), `beta = 0.5` and the `k` range (4 to 16) were chosen by hand-tuning on this dataset. They are good starting points, not universal constants.

## Limitations

- **Memory capacity.** A Hopfield network reliably stores about `0.14 × N` patterns, which is roughly 6 for 49 neurons. Five patterns are used. Adding more, especially similar ones, will quickly degrade recall.
- **Similar letters.** A and H share many pixels, so they are the most likely to be confused under heavy noise.
- **Spurious states.** Hopfield networks can settle into mixtures of memories. The REJECT verdict exists partly to catch these.
- **Synthetic data.** Noise is simulated, not collected from a real sensor. Results on real scans would differ.
- **No tuning split.** The tuned hyperparameters were checked on the same type of noise used in the final experiment, so the numbers are indicative, not a rigorous benchmark.

## Customising the project

| To change | Edit |
|---|---|
| Stored letters | the `LETTERS` dictionary (7 strings of 7 characters each) |
| Grid size | `SHAPE`, plus the pattern strings |
| Noise levels tested | the list in the experiment loop in `main()` |
| Trials per level | `trials` in `main()` |
| Fuzzy "ON" definition | `a` and `c` in `s_membership()` |
| Evidence strength | `beta` in `recall_fuzzy()` |
| Softness schedule | `k_start`, `k_end` in `recall_fuzzy()` |
| Verdict thresholds | the `conf` cut-offs in `fuzzy_decide()` |
| Fuzzy rule shapes | the trapezoid parameters in `fuzzy_decide()` |

## Ideas for extension

- Test the memory capacity by adding more patterns and plotting accuracy against the number stored.
- Replace the hand-set rule base with a larger fuzzy inference system.
- Use a pseudo-inverse (projection) learning rule and compare it with Hebbian learning.
- Add a Gaussian or bell membership option and compare it with the S-function.
- Generate a plot of accuracy versus noise (for example with matplotlib) for the report.

## File layout

```
.
├── fuzzy_hopfield.py   # the whole project (data, pre-processing, network, fuzzy layer, experiment)
└── README.md           # this file
```
