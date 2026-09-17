# Post-ICCV2019 Cardiac Anomaly-Detection — Consolidated Experiment Summary

*Last updated 2026-07-28 (folds in the 2-D era: jobs 3422342–3424567, 3466889/90, 3468535, 3492615, 3493632/3, 3516154/5).*

Everything built on top of the original **ICCV 2019 appearance–motion GAN** (surveillance-video
anomaly detection) after repurposing it for **cardiac MRI**. All methods train on **healthy (NOR)
data only** and flag disease as an anomaly score.

**How to read every number below.** Metric = **patient-level AUC (NOR vs disease)**. Two axes
structure *all* results:

| Axis | Dataset | What it measures |
|---|---|---|
| **ACDC** | single-vendor | in-distribution disease detection (the *easy* axis) |
| **M&Ms** | multi-vendor / multi-scanner | **cross-vendor generalisation — the wall this project keeps hitting** |

The entire research arc is one question: **can an unsupervised, NOR-only detector cross the
M&Ms multi-vendor domain gap?** ACDC is nearly solved; M&Ms is where methods live or die.

**Power (read this before believing any single cell).** The held-out sets are
**ACDC-50 = 10 NOR / 40 disease** and **M&Ms-Test = 32 NOR / 104 disease**. Bootstrap 95% CIs are
roughly **±0.16 on ACDC** and **±0.09 on M&Ms**. Differences smaller than that are *not measurable* —
read every table for **direction and consistency across cells**, never for a single winning number.
Model selection was done on **M&Ms-Validation (9 NOR)**, which is weaker still; where a claim
depends on selection, the honest val-selected number is reported alongside.

---

## Master results table

Headline patient-Mean AUC per method (best scorer / stream). Grouped by family.
Rows 1–14 are the pre-2-D era; rows 16–20 are the current state and **supersede** them where they
overlap.

| # | Method | Family | ACDC | M&Ms | Verdict |
|---|---|---|---|---|---|
| 0 | ICCV2019 GAN, Flow-SSIM (cardiac port) | GAN recon | **0.8125** | **0.7310** | **the bar** — honest held-out |
| 1 | CineMA-faithful frozen probe | density | 0.761 | 0.590 | ACDC clears bar; **M&Ms gap** |
| 2 | CineMA legacy-2D probe | density | 0.617 | 0.508 | off-distribution; weak |
| 3 | **DINOv2** frozen probe (sweep peak) | density | **0.735** | **0.704** | first appearance encoder to cross M&Ms |
| 4 | MAE (ImageNet) frozen probe | density | 0.678 | 0.521 | inherits the gap |
| 5 | LoRA adapter (Option A) | adaptation | 0.79 | 0.59 | **negative** — M&Ms flat |
| 6 | QFAE-CineMA, appearance-only | QFAE recon | 0.845 | 0.549 | best in-distn appearance; MM unmoved |
| 7 | QFAE-CineMA dual, motion stream (Farneback) | QFAE recon | 0.639 | 0.716 | ⚠ **9-NOR noise** — see ★, reads 0.636 held-out |
| 8 | QFAE-CineMA dual, appearance (Farneback) | QFAE recon | 0.909 | 0.508 | flow head boosts ACDC |
| 9 | QFAE-CineMA dual, combined (Farneback) | QFAE recon | 0.779 | 0.598 | 0.713 overall |
| 10 | QFAE-CineMA dual, combined (registration flow) | QFAE recon | **0.936** | 0.554 | best ACDC of the project (pre-2-D) |
| 11 | Percentile-rank fusion (offline, on #9) | fusion | — | — | ⚠ pooled metric; **retired**, see ★★.6 |
| 12 | DINOv2-QFAE, coupled scorer | QFAE recon | 0.642 | 0.495 | reconstruction can't read DINOv2 |
| 13 | DINOv2-QFAE, paper-faithful (separate MAE scorer) | QFAE recon | 0.798 | 0.370 | ACDC fixed; MAE scorer vendor-blind |
| 14 | Masked-motion QFAE (MGMAE/MME) | QFAE recon | 0.777 | 0.693 | held-out; best pre-2-D QFAE M&Ms stream |
| 15 | MTL-MAD | assessment only | — | — | borrow MoE + rank-fusion machinery, not tasks |
| **16** | **2-D QFAE, MAE@224 — flow_SSIM, middle-60%** | QFAE recon | **0.843** | **0.732** | **current best balanced model — GAN parity** |
| **17** | **2-D QFAE, MAE@224 — mag_SSIM, middle-60%** | QFAE recon | **0.860** | **0.743** | **best held-out pair in the project** (but val can't pick it) |
| 18 | 2-D QFAE, MAE@224 — flow_L1, middle-60% | QFAE recon | **0.865** | 0.697 | best ACDC motion; loses M&Ms |
| 19 | 2-D QFAE, DINOv2@224 — mag_SSIM, middle-60% | QFAE recon | 0.782 | 0.736 | second-best M&Ms |
| 20 | 2-D QFAE, MAE@224 — appearance (any form) | QFAE recon | 0.56–0.73 | 0.44–0.60 | **dead weight** — see ★★.5 |
| 21 | 2-D QFAE, MAE@224 reg-flow + appearance (fusion) | QFAE recon | **0.874** | 0.721 | reg-flow's in-distn/fusion win — see ★★.9 |
| 22 | 2-D QFAE, DINOv2@224 pixel-MAE + flow (fusion) | QFAE recon | 0.850 | 0.730 | best appearance-score fusion — see ★★.10 |
| 23 | 2-D QFAE, DINOv3@224 — mag_SSIM, middle-60% | QFAE recon | **0.873** | 0.721 | best ACDC motion encoder; M&Ms ≈ MAE (within CI) — see ★★.11 |

**Best-of (held-out, 2026-07-28):** ACDC → **0.890** (matrix `mae224×mae`, middle-60%);
M&Ms → **0.743** (#17). **Best balanced single detector → 2-D QFAE MAE@224 motion stream**
(≈0.84–0.89 ACDC / 0.71–0.74 M&Ms), which **matches the GAN on M&Ms and beats it on ACDC** —
but see ★★.7: nothing separates them at this sample size, and honest val-selection lands *below*
the GAN.

---

## ★ Honest held-out evaluation (2026-07-22) — the protocol fix

Earlier rows compared a **pooled** GAN number (`0.73–0.77`, ~49% cross-dataset ranking pairs =
partly a vendor classifier) against **per-dataset** QFAE numbers on **M&Ms-Validation (9 NOR)** —
not apples-to-apples, and underpowered. Fixed here: every model scored **patient-Mean AUC, per
dataset, on identical patients**, selecting on M&Ms-Validation and reporting on held-out
**ACDC-50 + M&Ms-Testing (136 pts: 32 NOR / 104 disease)**. `qfae_report.py` builds the tables
(bootstrap 95% CIs). **This protocol is used for every number in this document from here on.**

| Model / stream | ACDC (50) | **M&Ms-Test (136, held-out)** [95% CI] |
|---|---|---|
| **GAN flow-SSIM, patient_mean** *(the bar)* | **0.8125** | **0.7310** |
| QFAE Farneback — flow_SSIM | 0.650 | 0.636 [0.52, 0.74] |
| QFAE Farneback — flow_L1 | 0.892 | 0.713 [0.61, 0.81] |
| QFAE reg-flow — flow_SSIM | 0.877 | 0.650 [0.54, 0.75] |
| QFAE reg-flow — combined | 0.940 | 0.601 [0.49, 0.71] |
| QFAE masked-motion — flow_SSIM | 0.777 | 0.693 [0.59, 0.79] |
| QFAE appearance (any variant) | 0.90–0.94 | 0.52–0.56 |

Findings that survived: **"motion crosses the gap at 0.716" was 9-NOR noise** (0.711 val → 0.636
test); **masking genuinely helped** (0.636 → 0.693); **appearance stays vendor-blind** (0.52–0.56);
the GAN won the cross-vendor axis against every pre-2-D QFAE stream.

---

## ★★ The 2-D era (2026-07-24 → 2026-07-28) — the current state of the project

Three experiments in sequence. Together they change the headline model and close out the
score-side and training-side levers.

### ★★.1 What changed methodologically

1. **Single-pass ED input** (`--single_pass`, `single_pass_ed=True`): one forward pass on the ED
   frame instead of the ED/ES two-pass feed. Jobs 3422342/3.
2. **Genuinely 2-D encoders**: DINOv2 / ImageNet-MAE run per slice at native 2-D, with **no slice
   token and no depth-16 replication** (`qfae_dino_train.py` / `qfae_dino_eval.py`), on
   CineMA-faithful preprocessing. Jobs 3422656–3422661, val eval 3424567.
3. **Middle-60% slice aggregation** (`middle60`): drop the top/bottom 20% of slices before the
   patient-level reduction — the same rule the GAN's loaders already used.

**This reverses the earlier "CineMA is the best motion encoder" finding.** That result was an
artifact of forcing 2-D encoders through CineMA's 3-D depth-16 container, where the encoder sees
16 replicated identical slices — off-distribution for its depth-wise attention.

### ★★.2 Frozen-encoder comparison (`summarize_encoders.py`) — held-out ACDC-50 / M&Ms-Test

| encoder | appearance (mean) | flow_SSIM (mean) | flow_SSIM (mid60) | mag_SSIM (mid60) |
|---|---|---|---|---|
| CineMA (single-pass, 3-D) | 0.835 / 0.542 | 0.795 / 0.689 | 0.820 / 0.727 | 0.770 / 0.716 |
| DINOv2 @224 (2-D) | 0.585 / 0.605 | 0.775 / 0.718 | 0.790 / 0.721 | 0.782 / **0.736** |
| DINOv2 @518 (2-D) | 0.650 / 0.589 | 0.795 / 0.694 | 0.823 / 0.708 | 0.835 / 0.711 |
| **MAE @224 (2-D)** | 0.562 / 0.548 | 0.812 / 0.714 | **0.843 / 0.732** | **0.860 / 0.743** |
| DINOv3 @224 (2-D) | 0.665 / 0.593 | 0.805 / 0.695 | **0.858** / 0.712 | **0.873** / 0.721 |

Motion streams under middle-60% (`confirm_offline.py`):

| model | flow_SSIM | mag_SSIM | flow_L1 |
|---|---|---|---|
| CineMA-SP | 0.820/0.727 | 0.770/0.716 | 0.810/0.688 |
| DINOv2@224 | 0.790/0.721 | 0.782/**0.736** | 0.823/0.669 |
| DINOv2@518 | 0.823/0.708 | 0.835/0.711 | 0.860/0.674 |
| **MAE@224** | 0.843/**0.732** | **0.860/0.743** | **0.865**/0.697 |
| DINOv3@224 | **0.858**/0.712 | **0.873**/0.721 | **0.873**/0.673 |

### ★★.3 Middle-60% aggregation is the single biggest free lever

Middle-60% is the top reduction rule in **11/12** (model × motion-stream) cells on ACDC and
**8/12** on M&Ms. It costs nothing, needs no tuning, and lifts MAE@224 flow_SSIM from
0.812/0.714 → **0.843/0.732**. Apical and basal slices are where flow is least reliable
(through-plane motion, partial volume); dropping them is a physiology-motivated denoiser.

### ★★.4 Encoder × scorer matrix (job 3468535, `qfae_matrix_report.py`) — the scorer is irrelevant

Full 3×3: `{CineMA, DINOv2@224, MAE@224}` encoder × `{CineMA, DINOv2, MAE}` perceptual scorer.
Cells are ACDC / M&Ms-Test.

**MOTION — flow_SSIM / mean** (pre-committed stream+rule):

| encoder \ scorer | cinema | dino | mae |
|---|---|---|---|
| cinema | 0.795/0.689 | 0.823/0.679 | 0.802/0.682 |
| dino224 | 0.777/0.709 | 0.777/0.712 | 0.795/0.705 |
| **mae224** | 0.835/0.710 | 0.840/**0.726** | **0.845**/0.709 |

**MOTION — flow_SSIM / middle-60%** (secondary rule):

| encoder \ scorer | cinema | dino | mae |
|---|---|---|---|
| cinema | 0.820/0.727 | 0.853/0.715 | 0.818/0.717 |
| dino224 | 0.792/0.727 | 0.810/0.733 | 0.797/0.725 |
| **mae224** | 0.877/0.724 | 0.877/**0.735** | **0.890**/0.725 |

**RECONSTRUCTION — appearance / mean**:

| encoder \ scorer | cinema | dino | mae |
|---|---|---|---|
| cinema | 0.835/0.542 | 0.762/0.567 | 0.720/0.603 |
| dino224 | 0.693/0.507 | 0.635/0.609 | 0.640/0.558 |
| mae224 | 0.710/0.467 | 0.713/0.602 | 0.730/0.532 |

**Conclusions from the matrix:**
1. **The perceptual scorer does not matter for motion.** Within-row M&Ms spread is **≤ 0.017**
   against a ±0.09 CI. The coupled-vs-decoupled distinction that mattered for DINOv2 in the
   pre-2-D era (row 12 vs 13) is **dead** in the 2-D regime. So `MAE224 × MAE224` is not a
   meaningful pairing — the **encoder row** is the lever, the scorer column is noise.
2. **The MAE@224 row is the best row**, leading ACDC by 0.04–0.07 over every other encoder while
   holding M&Ms at 0.705–0.735.
3. **Appearance is vendor-limited in all nine cells** (M&Ms 0.44–0.62), regardless of encoder or
   scorer. There is no encoder×scorer combination that makes reconstruction cross the gap.
4. **M&Ms motion is remarkably stable at 0.705–0.735 across all nine cells** — consistent with a
   ~0.73 ceiling for this signal, matching the GAN exactly.

*Matrix caveats:* rows use different containers (CineMA 3-D vs 2-D per-slice) **and** different
batch sizes, so compare **within** a row. A regression check reproduced two known cells exactly
(`mae224×mae` BS=32 → 0.812/0.714; `cinema×cinema` → 0.795/0.689). Note the same cell at BS=8
reads 0.845/0.709 vs BS=32 0.812/0.714 — i.e. **run-to-run/hyperparameter jitter is ~±0.03 on
ACDC**, another reason not to over-read single cells. `top_frac=0.2` is a fraction of each
scorer's own token count, so absolute pooled-token counts differ by scorer (CineMA 2304→461,
p14 256→51, p16 196→39).

### ★★.5 Flow-head ablation + pixel-space appearance (jobs 3466889/90, `analyze_flowabl.py`)

**Part A — does the appearance head help or hurt the flow head?** Sweep `λ_appe ∈ {1.0, 0.5, 0.25, 0}`
on MAE@224, reporting flow_SSIM alone:

| λ_appe | mean (ACDC/M&Ms) | middle60 (ACDC/M&Ms) |
|---|---|---|
| 1.0 (baseline) | 0.812/0.714 | 0.843/0.732 |
| 0.5 | 0.815/0.717 | 0.853/0.734 |
| 0.25 | 0.805/0.719 | 0.853/0.731 |
| 0.0 (flow-only) | 0.805/0.716 | 0.848/0.729 |

**The appearance head is neutral for the flow stream** — flow-only ≡ joint training within
Δ ≤ 0.003 on M&Ms. So **0.843/0.732 is the flow head's ceiling**, not a number that joint
training is suppressing. The training-side lever is exhausted.

**Part B — is GAN-style *pixel* appearance a better stream than embedding appearance?**

| model | rule | ACDC pixel/embed | M&Ms pixel/embed |
|---|---|---|---|
| MAE@224 | mean | 0.608 / 0.562 | 0.592 / 0.548 |
| MAE@224 | middle60 | 0.585 / 0.578 | 0.577 / 0.558 |
| DINOv2@224 | mean | 0.647 / 0.585 | 0.616 / 0.605 |
| DINOv2@224 | middle60 | 0.625 / 0.557 | 0.601 / 0.612 |

Pixel-space appearance is **slightly stronger** than embedding appearance (+0.03–0.06) — the GAN's
choice was the better one — but it is **still vendor-limited** (0.58–0.62 on M&Ms) and **still
doesn't fuse** (below).

### ★★.6 Fusion in the 2-D regime — appearance never adds (`fusion_sweep_2d.py`)

Percentile-rank *and* z-score normalisation, equal-weight fusion, per dataset, held-out:

| model / rule | appe alone | flow alone | equal-weight fuse | oracle best-w | val-selected-w |
|---|---|---|---|---|---|
| MAE@224 / mean | 0.548 | **0.714** | 0.673 | 0.714 (w=0.00) | 0.714 (w=0.00) |
| MAE@224 / middle60 | 0.558 | **0.732** | 0.706 | 0.741 (w=0.20) | 0.737 (w=0.10) |
| DINOv2@224 / mean | 0.605 | **0.718** | 0.692 | 0.718 (w=0.10) | 0.671 (w=0.65) |
| CineMA-SP / mean | 0.542 | **0.689** | 0.644 | 0.689 (w=0.00) | 0.688 (w=0.05) |

*(M&Ms-Test column; w = weight on appearance.)* **Equal-weight fusion always hurts M&Ms**, and the
**oracle weight is w ≈ 0–0.2** — i.e. even peeking at test labels, the best thing to do with the
appearance stream is nearly to discard it. Same story with pixel-space appearance. On ACDC fusion
does help CineMA-SP (0.795 → 0.864–0.882), which is the one place appearance carries signal.

**This retires row 11.** The old "rank fusion 0.713 → 0.754" win was measured on the **pooled**
metric (~49% cross-dataset pairs); per-dataset, fusion is neutral-to-harmful on the axis that
matters.

### ★★.7 The honest val-selection reality check (`confirm_val.py`)

Select the motion recipe on **M&Ms-Val (9 NOR)**, report on **M&Ms-Test (32 NOR)**:

| model | picked on val | val M&Ms | **test M&Ms** | test ACDC |
|---|---|---|---|---|
| CineMA-SP | flow_SSIM / top20 | 0.658 | 0.627 | 0.680 |
| DINOv2@224 | flow_SSIM / mean | 0.684 | 0.718 | 0.775 |
| DINOv2@518 | flow_SSIM / mean | 0.676 | 0.694 | 0.795 |
| **MAE@224** | flow_SSIM / mean | **0.702** | **0.714** | 0.812 |

**Global pick across all configs → MAE@224 / flow_SSIM / mean → held-out M&Ms-Test 0.714
(ACDC 0.812), vs GAN 0.731.** Across the 3×3 matrix the global val pick is `dino224×mae` → **0.705**.

**Two honest caveats on the headline:**
- **Val cannot pick the best rule.** `middle60` is never selected on the 9-NOR val set (it picks
  `mean`/`top20`), so the 0.843/0.732 and 0.860/0.743 numbers are **test-set reads**, not
  val-selected results. Concretely: MAE@224 `mag_SSIM/middle60` has **val 0.631 → test 0.743** —
  the val set is blind to the project's best configuration.
- **Under honest selection the QFAE lands at 0.705–0.714, below the GAN's 0.731** (well inside the
  ±0.09 CI, so "not significantly worse", but not a win).

### ★★.8 What the current best model actually is

**2-D QFAE, frozen MAE@224 encoder, motion (flow_SSIM / mag_SSIM) stream, middle-60% aggregation.**
The perceptual scorer can be MAE, DINOv2, or CineMA — it moves nothing. The appearance head can be
removed entirely (λ_appe=0) with no loss.

- **Defensible claim:** it **matches the GAN cross-vendor** (0.732–0.743 vs 0.731) and **clearly
  beats it in-distribution** (0.843–0.890 vs 0.8125).
- **Not defensible:** that it beats the GAN on M&Ms. CIs overlap completely, and honest
  val-selection puts it at 0.705–0.714.
- **Honest headline detector for the cross-vendor axis remains the GAN Flow-SSIM at 0.731.**

### ★★.9 Registration-flow teacher for the 2-D encoders (jobs 3493632/3, `analyze_regflow.py`)

Trained MAE@224 and DINOv2@224 (coupled scoring; everything else identical to the Farneback 2-D
models) with the biomechanics `Registration_Net` as the flow-GT teacher **instead of Farneback
optical flow** — a clean reg-vs-optical-flow comparison. Held-out ACDC-50 / M&Ms-Test, `F→R`
(Farneback → registration):

| stream / rule | MAE@224 ACDC | MAE@224 M&Ms | DINOv2@224 ACDC | DINOv2@224 M&Ms |
|---|---|---|---|---|
| flow_SSIM / mean | 0.812→0.838 | 0.714→0.703 | 0.775→0.818 | 0.718→0.669 |
| flow_SSIM / mid60 | 0.843→0.775 | 0.732→0.687 | 0.790→0.750 | 0.721→0.672 |
| mag_SSIM / mid60 | 0.860→0.688 | 0.743→0.681 | 0.782→0.688 | 0.736→0.678 |
| **appe+flow / mean** | 0.772→**0.874** | 0.673→**0.721** | 0.731→0.746 | 0.692→0.666 |

**Reg-flow reproduces the 3-D CineMA split (row 10) in the 2-D encoders:**
1. **Cross-vendor (M&Ms): reg-flow is worse than Farneback on every pure-motion stream** — it does
   not help the vendor gap.
2. **In-distribution (ACDC), `mean` rule: reg-flow helps** (MAE 0.812→0.838, DINOv2 0.775→0.818), but
   *loses* under middle-60% — the reg field is smoother (96 px internal), so dropping apex/base slices
   gives it less than it gave noisy Farneback.
3. **The win is fusion.** Reg-flow motion is more complementary to appearance:
   **MAE@224 reg-flow + appearance (equal-weight rank, mean) = ACDC 0.874 / M&Ms 0.721** — the
   best-balanced 2-D result, beating the GAN on ACDC and ~matching it on M&Ms, and clearly above the
   Farneback fusion (0.772/0.673).

**Verdict:** reg-flow is an **in-distribution + fusion lever, not a cross-vendor one** — the same
conclusion as 3-D CineMA. For *pure* cross-vendor motion, **Farneback + middle-60% remains best**
(MAE mag_SSIM 0.743). Consistent with conclusion 2: M&Ms motion saturates at ~0.73 regardless of the
flow teacher.

### ★★.10 Pixel score-way sweep — the appearance *score metric* matters (job 3492615, `analyze_pixscore.py`)

Re-scored the pixel-trained MAE@224 / DINOv2@224 reconstructions (**no retraining** — the decoder
output is fixed) with MSE, MAE(L1), 1−SSIM and MSE+grad, mirroring the GAN notebook's multi-measure
appearance sweep. Held-out, rule = `mean` (mean > middle60 for appearance — the middle-60% lever is
motion-specific):

| encoder | appe MSE | appe MAE(L1) | appe 1−SSIM | appe MSE+grad |
|---|---|---|---|---|
| MAE@224 | 0.632/0.607 | 0.688/0.596 | 0.657/0.515 | 0.608/0.592 |
| DINOv2@224 | 0.682/0.649 | **0.725/0.665** | 0.713/0.526 | 0.647/0.616 |

1. **The score metric matters as much as the encoder.** MAE(L1) is the best pixel appearance score;
   MSE+grad (the default) is middling; **1−SSIM is the *worst* on M&Ms (0.51–0.53)** — a trained AE
   reproduces global structure regardless of vendor, so SSIM washes out the anomaly residual. The
   SSIM lever is motion-specific, not transferable to appearance.
2. **Strongest appearance stream in the project: DINOv2@224 pixel-MAE = ACDC 0.725 / M&Ms 0.665**
   (> embedding 0.605, > MSE+grad 0.616).
3. **First appearance stream strong enough to fuse without hurting M&Ms:** DINOv2 pixel-MAE + flow
   (equal-weight rank) = **ACDC 0.850 / M&Ms 0.730** (oracle 0.733, w=0.35). This **nuances ★★.6 /
   conclusion 7**: with a better appearance score, DINOv2 fusion is net-neutral-to-slightly-positive
   on M&Ms (+0.012, within CI, and val can't lock the weight) and a clear ACDC gain — still no robust
   break past 0.73.

### ★★.11 DINOv3 ViT-B/16 encoder (jobs 3516154/5, `summarize_encoders.py`)

Added **DINOv3 ViT-B/16** (`vit_base_patch16_dinov3.lvd1689m`) as a frozen encoder — same 2-D recipe as
MAE@224 (both heads, coupled scoring, Farneback flow), a pure drop-in (768-d, patch16, 4 register tokens
auto-handled; needed only a `timm 1.0.15 → 1.0.28` upgrade in `derisk`). Held-out (each cell ACDC / M&Ms):

| stream | MAE@224 | DINOv3@224 |
|---|---|---|
| appearance (mean) | 0.562/0.548 | **0.665**/0.593 |
| flow_SSIM (mid60) | 0.843/**0.732** | **0.858**/0.712 |
| mag_SSIM (mid60) | 0.860/**0.743** | **0.873**/0.721 |

**DINOv3 is the best *in-distribution* (ACDC) encoder** — top motion (mid60 0.858/0.873, edging MAE) *and*
better appearance (0.665, semantic like DINOv2's 0.585). But **on cross-vendor M&Ms it is slightly *below*
MAE** (motion 0.712–0.721 vs 0.732–0.743; well within ±0.09 CI, so statistically equal). Appearance stays
vendor-limited (0.593). DINOv3 is the **5th independent encoder to land M&Ms motion at 0.71–0.74** — the
~0.73 ceiling holds; a newer/stronger self-supervised backbone does not cross the vendor wall. Confirms the
**semantic (DINOv2/v3 → ACDC + appearance) vs texture (MAE → M&Ms motion)** split. MAE@224 remains the
best-balanced / best cross-vendor encoder.

---

## The families, in order

### 0. Original ICCV2019 GAN (cardiac port) — the baseline
Appearance-motion U-Net GAN (`GAN_tf.py`). Reconstruction + optical-flow aux head + PatchGAN.
Its **Flow-SSIM** stream (patient_mean, middle-60% slices) is the bar: **ACDC 0.8125 / M&Ms-Test
0.731**, held-out on 31–32 NOR. First evidence that **motion** carries the signal — a finding every
later family reproduced.

### 1. Frozen foundation-model probes (density: Ledoit-Wolf Mahalanobis / kNN on NOR features)
No decoder training — *"how far are a frame's frozen features from the NOR distribution?"*
(`derisk_cinema.py --backbone {cinema,dino,mae}`).
- **CineMA-faithful** (canonical 3-D SAX): ACDC **0.761** / M&Ms **0.590**.
- **DINOv2**: signal is **layer-separated** — mid layer (L5) → ACDC 0.735, deep layer (L11, @518px)
  → M&Ms **0.704**. First appearance encoder to cross the vendor gap via a *density* head.
- **MAE** (ImageNet): ACDC 0.678 / M&Ms 0.521 — inherits the gap. MAE is texture-biased and scanner
  texture *is* the vendor gap.
- Note the inversion in the 2-D era: MAE is the **worst appearance** encoder but the **best motion**
  encoder. The two roles are unrelated.

### 2. Label-free adaptation — LoRA adapter (Option A), **concluded negative**
NOR-only LoRA on CineMA via its own MAE loss (`adapt_cinema.py`, `cinema_lora.py`). ACDC 0.76 → 0.79,
M&Ms **flat at ~0.59** across 3 configs incl. all-frames (13× data). **The M&Ms gap is fundamental
to label-free adaptation.**

### 3. Q-Former Autoencoder (QFAE), pre-2-D — reconstruct through a Q-Former bottleneck
`qfae_cinema.py` + `qfae_train.py` + `qfae_eval.py` (arXiv 2507.18481).
- CineMA appearance-only: ACDC 0.845 / M&Ms 0.549.
- Dual-stream (+ flow head): appearance ACDC 0.909; the motion stream's apparent M&Ms 0.716 did not
  survive the powered re-eval (0.636).
- Registration-flow variant: ACDC 0.936 (best pre-2-D ACDC) / M&Ms 0.554.
- Q-Former defect: the real problem is **grid alignment**, not query count (784 decoder queries vs
  256 encoder tokens, 28×28 vs 16×16).

### 4. Masked autoencoders throughout
- **CineMA is itself a ConvMAE** — the frozen prior in the probes, the QFAE encoder, and the LoRA
  target. Its M&Ms ceiling (~0.59 appearance) is the anchor.
- **Masked-motion QFAE** (`qfae_masking.py`, MGMAE/MME-style input masking on the motion head):
  flow_SSIM 0.636 → **0.693** held-out — real, and the best pre-2-D QFAE M&Ms stream, still below
  the GAN. Its justification is that the QFAE port's `n_queries == 2304` removes the paper's
  bottleneck regulariser, which masking restores.

### 5. DINOv2-QFAE (pre-2-D) — **negative for M&Ms, instructive**
- Coupled (DINOv2 encodes and scores): ACDC 0.642 / M&Ms 0.495 — the decoder minimises the same
  distance it's scored on.
- Paper-faithful (separate frozen MAE scorer): ACDC **0.798** / M&Ms **0.370** — the MAE scorer is
  vendor-blind.
- **Superseded in part by ★★.4:** in the genuinely-2-D regime the scorer choice stops mattering for
  motion. The coupled-scorer collapse was specific to the 3-D-forced setting.

### 6. MTL-MAD — assessed, not run
Multi-task MoE anomaly detector (arXiv 2605.05891). Borrow the **machinery** (MoE multi-head +
rank fusion), not the **task list** (all appearance-domain → would inherit the M&Ms gap). Note the
fusion half of that recommendation is now weakened by ★★.6.

### 7. 2-D QFAE — the current best family
See **★★** above. `qfae_dino_train.py` / `qfae_dino_eval.py`, driven by `qfae_matrix.pbs`;
offline tables via `qfae_matrix_report.py`, `confirm_offline.py`, `confirm_val.py`,
`summarize_encoders.py`, `fusion_sweep_2d.py`, `analyze_flowabl.py`.

---

## Cross-cutting conclusions

1. **ACDC is essentially solved; M&Ms is the wall.** Nearly every method reaches ACDC 0.76–0.94.
   The whole story is M&Ms.
2. **Motion is the only mechanism that reliably crosses the vendor gap, and it saturates at ~0.73.**
   Optical flow is physiological and intensity/contrast-invariant, hence scanner-independent.
   Across 4 encoders × 3 scorers × 3 motion streams × 2 rules, every well-formed motion configuration
   lands in **0.69–0.74** on M&Ms-Test. The GAN (0.731) and the best 2-D QFAE (0.732–0.743) are the
   same number. **This looks like a ceiling of the signal, not of any one architecture.**
   (DINOv2 *feature density* reached 0.704 once, but reconstruction cannot surface it — see #5.)
3. **Every appearance/texture/reconstruction signal sits at 0.44–0.62 on M&Ms**, in every encoder,
   every scorer, embedding-space or pixel-space. Appearance is the vendor gap.
4. **The perceptual scorer is irrelevant for the motion stream** (within-row spread ≤ 0.017). Only
   the **encoder** matters, and only for how well it supports flow decoding — MAE@224 is the best
   motion encoder despite being the worst appearance encoder.
5. **The container was a confound, not a result.** "CineMA is the best motion encoder" was an
   artifact of forcing 2-D encoders through a 3-D depth-16 replication. Always check the input
   container before attributing a gap to the encoder.
6. **Aggregation beats architecture.** Middle-60% slice selection — free, tuning-free — is worth
   ~+0.02–0.03 on both axes and is the top rule in 11/12 ACDC and 8/12 M&Ms cells. It outperformed
   every architectural change tried in the same period.
7. **Both the score-side and training-side levers are essentially exhausted.** Fusion: appearance
   almost never adds (oracle w ≈ 0–0.2, equal-weight usually hurts). Training: λ_appe = 0 ≡ λ_appe = 1
   within 0.003. Two partial refinements found since: a *better appearance score* (DINOv2 pixel-MAE,
   ★★.10) and the *reg-flow motion teacher* (★★.9) both make appearance+flow fusion net-neutral on
   M&Ms and a clear ACDC win (0.850–0.874 ACDC / 0.721–0.730 M&Ms) — but neither breaks the ~0.73
   cross-vendor ceiling.
8. **Unsupervised adaptation cannot close M&Ms** (LoRA, 3 configs). Closing it likely needs
   supervised/few-shot signal.
9. **Selection is the binding constraint on claims, not modelling.** M&Ms-Val has 9 NOR and cannot
   pick the project's best configuration (`mag_SSIM/middle60`: val 0.631 → test 0.743). Any future
   headline needs a larger selection set, or it is a test-set read.

## Status ledger

| Line | Status |
|---|---|
| ICCV2019 GAN Flow-SSIM baseline | **established** — ACDC 0.8125 / M&Ms-Test 0.731 |
| CineMA frozen probe | done (0.761/0.590) |
| LoRA adapter (Option A) | **concluded — negative** |
| QFAE-CineMA (appearance / dual-stream / reg-flow) | done; ACDC 0.936 pre-2-D, M&Ms claims superseded |
| Percentile-rank fusion | **retired** — pooled-metric artifact; per-dataset fusion is neutral-to-harmful |
| Encoder-swap probe (DINOv2/MAE) + readout sweep | done (DINOv2 crosses gap via density) |
| DINOv2-QFAE (coupled + paper-faithful) | done — negative for M&Ms; ACDC 0.798 bankable |
| Masked-motion QFAE | **done** — 0.777/0.693 held-out; masking helps, still < GAN |
| 2-D single-pass QFAE (MAE/DINOv2) + middle-60% | **done — current best** (MAE@224 0.843–0.860 / 0.732–0.743) |
| Encoder × scorer 3×3 matrix | **done** — scorer irrelevant; appearance vendor-limited in all 9 cells |
| Flow-head ablation (λ_appe sweep) | **done** — appearance head neutral; 0.732 is the flow ceiling |
| Pixel-space appearance + 2-D fusion sweep | **done** — pixel slightly better, still doesn't fuse |
| Pixel score-way sweep (job 3492615) | **done** — MAE(L1) best appe score, 1−SSIM worst; DINOv2 pixel-MAE 0.725/0.665, fusion 0.850/0.730 |
| Reg-flow teacher for 2-D encoders (jobs 3493632/3) | **done** — in-distn/fusion lever; MAE@224 reg-flow+appe 0.874/0.721; Farneback still best pure M&Ms |
| DINOv3 ViT-B/16 encoder (jobs 3516154/5) | **done** — best ACDC encoder (0.858/0.873 mid60); M&Ms 0.712–0.721 ≈ MAE (within CI); ~0.73 ceiling holds (needed timm 1.0.28) |
| Multi-layer DINOv2 density probe | **proposed, not run** — the one untried M&Ms route |
| MTL-MAD | assessed only |

## Open questions

The motion route is saturated at ~0.73 on M&Ms and both its levers are spent. What remains:

1. **Multi-layer DINOv2 density probe** (mid L5 + deep L11, per-layer-whitened Mahalanobis) — the
   only untried route to M&Ms. DINOv2's 0.704 is a *density* property that reconstruction provably
   cannot surface (#5), so it needs the density head that worked in the probe.
2. **A vendor gate.** Streams are anti-correlated by dataset and a single global weight cannot own
   both axes; metadata is available and the oracle gated fusion sits ≈ 0.77.
3. **A larger selection set.** Per conclusion 9, M&Ms-Val (9 NOR) is too small to select the recipe
   that wins on test. Consider a NOR-only cross-validated selection over train+val, or reporting
   pre-committed recipes only.
4. **Accept the ceiling and reframe.** If ~0.73 is the cross-vendor limit of unsupervised motion
   anomaly detection, the contribution is the *ceiling itself* plus the two mechanisms that reach it
   — a defensible MRes result that the last four experiments all independently corroborate.
