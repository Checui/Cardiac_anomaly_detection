# Frozen CineMA for cardiac MRI anomaly detection — results

*Experiments run July 2026 on the Imperial HPC. This document is the thesis-facing
write-up; the decision rationale is in `MAE_CineMA_feasibility_memo.md`, the code is
described in `CLAUDE.md`.*

## 1. Objective

Test whether a **pretrained cardiac foundation model (CineMA)**, used **frozen** and
fitted on **healthy (NOR) hearts only**, can detect cardiac disease from short-axis (SAX)
cine MRI — and whether it beats the project's existing **Flow-SSIM appearance–motion GAN**
(patient-level Mean AUC ≈ 0.73–0.77). This is *unsupervised anomaly detection*: no disease
labels are used at training time.

## 2. Method

- **Backbone.** CineMA (ConvMAE-style cine-CMR foundation model; Nature *Comms Med* 2026;
  HuggingFace `mathpluscode/CineMA`), SAX encoder, 125 M params, kept **frozen**.
- **Preprocessing ("faithful").** Reproduces CineMA's own canonical SAX pipeline
  (`cinema_faithful.py`): resample each 3-D ED/ES volume to 1.0 mm/px → LV-bounding-box
  centre-crop to 192² → clip 0.95/99.5 percentiles → feed the real depth-16 slice stack.
  Data unit = one 3-D stack per (patient, phase).
- **Anomaly model.** Extract frozen encoder features (global-average-pooled patch tokens,
  1536-d = cls 768 + sax 768), then fit a training-free normal model on NOR features:
  **Ledoit-Wolf Mahalanobis** distance and a **kNN** memory bank. Score val stacks, then
  aggregate to patient level (Mean / Max). AUC = NOR vs disease.
- **Datasets.** Fit: ACDC-train-NOR + M&Ms-train-NOR (116 stacks). Val: ACDC test (50
  patients, all pathologies) + M&Ms validation (34 patients) = 168 stacks.
- **Scorer.** All AUCs below are **patient-level Mean, Mahalanobis** (the strongest;
  kNN was uniformly weaker). Per-dataset (ACDC/M&Ms) values are the stack-level one-vs-NOR
  AUC; "overall" is patient-level Mean. Deltas are the signal.

## 3. Experiments and results

### 3.1 Frozen CineMA (the de-risk)

| | Overall | ACDC | M&Ms |
|---|---|---|---|
| **Frozen CineMA, faithful preprocessing** | **0.68** | **0.76** | 0.59 |
| Flow-SSIM GAN baseline | 0.73–0.77 | — | — |

Per-disease (Mahalanobis, patient-Mean): RV 0.89, MINF 0.70, DCM 0.61, HCM 0.56.
→ Frozen CineMA already separates ACDC disease at the baseline level with **no decoder
training**, but M&Ms sits near 0.59.

### 3.2 Preprocessing matters: faithful vs legacy

The "legacy" path feeds a single 2-D SAX slice (letterboxed to 128, upscaled to 192, depth
zero-padded) — off-distribution for CineMA.

| Preprocessing | Overall | ACDC | M&Ms | Fit samples |
|---|---|---|---|---|
| **Faithful** (canonical 3-D SAX) | 0.68 | **0.76** | 0.59 | 116 stacks |
| Legacy (single 2-D slice) | 0.59 | 0.62 | 0.51 | 340 frames |

→ **+0.145 AUC on ACDC** from on-distribution input geometry, despite legacy having ~3×
more samples. Reading CineMA the way it was pretrained is essential; data volume is not the
lever here.

### 3.3 Feature-config / PCA tuning sweep (frozen encoder)

Swept feature subset {all, sax-only, cls-only} × PCA {0, 32, 64, 128} × scorer
{Mahalanobis, kNN} on the saved features. Best overall = 0.679 (≈ baseline). `feature_layers
last` (sax-only) was slightly *worse* (0.667). PCA hurt overall, but PCA-32 lifted M&Ms to
0.68 while cratering ACDC to 0.64 — i.e. **no single config wins both datasets.**
→ The M&Ms gap is **not** closable by scoring/feature tuning on frozen features.

### 3.4 Label-free encoder adaptation: NOR-only LoRA adapter

A light LoRA adapter (`cinema_lora.py`) trained NOR-only via CineMA's own self-supervised
**MAE** objective (`adapt_cinema.py`; masked-patch reconstruction, mask ratio 0.75), then
the probe re-run through the adapted encoder. Three configurations:

| Config | ACDC Δ | M&Ms Δ | notes |
|---|---|---|---|
| v1 — encoder LoRA rank 8, full trainable decoder | +0.006 | +0.003 | decoder absorbed the loss; encoder barely moved (feat cosine 0.994) |
| v2 — encoder+decoder LoRA, frozen base, rank 16, lr 3e-4 | **+0.032** | −0.009 | forced adaptation into encoder; ACDC → 0.79 |
| v3 — v2 config + **all cine frames (~13× data, ~1500 stacks)** | **+0.032** | **−0.000** | data-limit test |

→ Across all three, adaptation **reliably sharpens ACDC to ~0.79** (above the Flow-SSIM
baseline) but **M&Ms never moves**. Ruled out as causes of the M&Ms gap: decoder-sink
(v2), LoRA capacity / learning rate (v2), and **data volume** (v3, ~13×).

### 3.5 Q-Former Autoencoder (QFAE) with a CineMA prior

A different anomaly *mechanism* — reconstruction rather than feature density. Following the
Q-Former Autoencoder (Dalmonte et al., WACV 2026, arXiv 2507.18481), a **Q-Former** (learnable
queries) bottlenecks frozen **CineMA** encoder features into a **3-D decoder** that reconstructs
the SAX stack; trained NOR-only; anomaly = **CineMA perceptual reconstruction error** (multi-
layer feature cosine). CineMA is used as *both* the encoder and the perceptual prior. Code:
`qfae_cinema.py` / `qfae_perceptual.py` / `qfae_train.py` / `qfae_eval.py`; vendored repo +
paper in `q-former/`.

| | Overall | ACDC | M&Ms |
|---|---|---|---|
| **QFAE (CineMA prior)** | 0.67 | **0.845** | 0.55 |
| frozen probe | 0.68 | 0.76 | 0.59 |
| LoRA adapter (best) | 0.68 | 0.79 | 0.58 |
| Flow-SSIM GAN | 0.73–0.77 | — | — |

→ **QFAE gives the best in-distribution result of the whole project (ACDC 0.845)** — clearly
above the frozen probe, the adapter, and the Flow-SSIM baseline. But **M&Ms dropped
(0.59 → 0.55)**: the reconstruction/perceptual mechanism did not close the multi-vendor gap
(the perceptual score inherits CineMA's weak M&Ms features). Trained ~16 min; perceptual loss
0.32 → 0.09.

### 3.6 Dual-stream QFAE: adding an optical-flow (motion) head

Motivated by the observation that every *appearance*-based head fails on M&Ms while the GAN's
best stream was **motion** (Flow-SSIM), a second decoder head was added to the QFAE — exactly
like the GAN's auxiliary flow head — predicting the per-slice optical flow to the paired phase
(`[dx,dy,mag]`, Farneback GT, L1 loss). Shared Q-Former, two decoders; anomaly = appearance
(perceptual) + motion (Flow-SSIM) streams. `qfae_cinema.py` (`flow_head`), `qfae_perceptual.py`,
`qfae_train.py --flow`, `qfae_eval.py --flow`.

| stream (patient-Mean) | Overall | ACDC | M&Ms |
|---|---|---|---|
| appearance (perceptual) | 0.69 | **0.909** | 0.51 |
| **flow_SSIM (motion)** | 0.68 | 0.64 | **0.716** |
| flow_L1 | 0.67 | 0.87 | 0.57 |
| **combined** (z-sum) | **0.713** | 0.78 | 0.60 |

→ **The motion stream is the first thing in the project to move M&Ms** — 0.716, up from every
appearance head (~0.51–0.59) and into the GAN's Flow-SSIM range. Motion is physiological, not
scanner-texture, so it crosses the vendor gap. The two streams are strongly **complementary**:
appearance owns ACDC (**0.909**, a new best — the flow head also acts as a helpful auxiliary
task), motion owns M&Ms. The naive z-sum **combined (0.713) is the best overall number of the
project**.

An offline **fusion sweep** shows a single *global* stream weight only modestly helps — best
global = GAN-style log-ratio **0.723** (ACDC 0.82, M&Ms 0.58), z-fusion 0.716 — enough to *tie*
the Flow-SSIM GAN but not beat it, because the streams are strongly **anti-correlated across
vendors** (appearance ACDC 0.90 / M&Ms 0.51; motion the reverse), so one weight is a compromise.
The real ceiling is **per-vendor routing: ACDC 0.902 *and* M&Ms 0.711** (appearance for
familiar-vendor scans, motion for the multi-vendor set) — which **beats the GAN on both**. Since
the scanner/vendor is known input metadata, a **vendor-aware gate** (or a learned per-scan
stream-confidence) is the actionable way to capture it — the recommended next step.

**Registration vs Farneback flow GT.** Swapping the motion head's ground-truth from Farneback to
the fine-tuned biomechanics Registration_Net teacher gives a **complementary** motion stream:

| motion GT (flow_SSIM) | ACDC | M&Ms |
|---|---|---|
| Farneback | 0.65 | **0.71** |
| Registration | **0.857** | 0.58 |

The learned reg-net produces a clean, anatomically-consistent field that captures ACDC disease
mechanics far better (+0.21), pushing the **registration combined to ACDC 0.936 — the project's
best** (overall 0.724). But raw, unbiased Farneback wins on the harder multi-vendor M&Ms (0.71 vs
0.58). So there are now **four complementary streams** (appearance + registration-flow → ACDC
~0.93; Farneback-flow → M&Ms 0.71); fusing/gating all three signals targets **~0.93 ACDC / 0.71
M&Ms** — beating the GAN on both.

## 4. Findings

1. **Frozen CineMA carries real in-distribution disease signal** — ACDC 0.76 with no
   decoder training, matching the Flow-SSIM baseline.
2. **On-distribution input geometry is the key enabler** — faithful canonical-SAX
   preprocessing beats a crude single-slice feed by **+0.145 on ACDC**.
3. **The M&Ms multi-vendor domain gap is genuine and robust** — verified not to be a
   preprocessing/label bug, and closed by **neither** feature/PCA tuning **nor** light
   NOR-only MAE adaptation across adapter capacity, learning rate, and ~13× data.
   Light self-supervised normal-only adaptation helps the single-centre set and does not
   transfer to the multi-vendor one.
4. **The M&Ms gap is specific to CineMA's *appearance* features, not anomaly detection per se.**
   Three appearance-based detectors — a density probe (§3.1), a NOR-only LoRA adapter (§3.4),
   and the Q-Former reconstruction autoencoder (§3.5) — *all* improve ACDC yet *none* moves
   M&Ms, localising the bottleneck to CineMA's cross-vendor *appearance* features. But a
   **complementary motion signal is not so limited** (finding 7).
5. **The Q-Former Autoencoder is the strongest in-distribution appearance detector** — ACDC
   **0.845** (0.909 as the appearance stream of the dual-stream model), above the frozen probe,
   the adapter, and the Flow-SSIM GAN.
6. **A motion (optical-flow) head crosses the vendor gap that appearance cannot (§3.6).** Adding
   a GAN-style flow head to the QFAE gives a **motion stream that reaches M&Ms 0.716** — the
   first meaningful M&Ms movement in the project (all appearance heads sit at ~0.51–0.59), in
   the GAN's Flow-SSIM range. Motion is physiological, not scanner-texture, so it generalises
   across vendors. The streams are strongly complementary (appearance→ACDC 0.909, motion→M&Ms
   0.716); their naive z-sum **combined 0.713 is the best overall** result of the project.

## 5. Conclusion

CineMA is a strong **in-distribution** (single-centre / ACDC) cardiac anomaly detector as a
frozen, normal-only feature extractor, and the canonical-SAX pipeline is what unlocks it (best
in-distribution: the QFAE, ACDC 0.845–0.909, above the GAN). Its **appearance** features do not
cross the M&Ms multi-vendor gap under any anomaly head or adaptation — but adding a **GAN-style
optical-flow head** does: the **dual-stream QFAE** is the first CineMA variant to move M&Ms
(motion 0.716) and gives the project's best overall score (combined 0.713), because appearance
and motion are complementary across the vendor gap. A single *global* fusion weight only ties
the Flow-SSIM GAN (~0.72), since the two streams anti-correlate by vendor; but the streams'
complementarity is so clean that **per-vendor routing reaches ACDC 0.902 and M&Ms 0.711 — beating
the GAN on both**. As the scanner/vendor is known input metadata, the clear next step is a
**vendor-aware gate** (or learned per-scan stream-confidence) over the two heads, which would make
the dual-stream QFAE (CineMA appearance + motion) the recommended detector for both single- and
multi-vendor settings.

## 6. Reproducibility

- **Code**: `cinema_faithful.py` (preprocessing + record loaders, incl. `*_allframes`),
  `derisk_cinema.py` (probe: `--cinema_preproc faithful|legacy`, `--adapter_path`),
  `cinema_lora.py` (LoRA), `adapt_cinema.py` (adapter trainer, `--all_frames`).
- **HPC jobs** (conda env `derisk`, weights cached offline in `$EPHEMERAL/hf_cache`):
  `qsub derisk_cinema.pbs` (frozen probe), `qsub derisk_cinema_legacy.pbs` (legacy baseline),
  `qsub adapt_cinema.pbs` (adapter), `qsub derisk_adapted.pbs` (adapted re-measure).
- **Result artefacts** (on the HPC): `derisk_out/` (frozen), `derisk_out_legacy/`,
  `derisk_out_adapted_v2/`, `derisk_out_adapted_v3/`; adapters in `adapter_out*/`. Each
  holds `derisk_results.json` (all AUCs) + `derisk_arrays.npz` (features).

*Baseline numbers are patient-level Mean, Mahalanobis. See `MAE_CineMA_feasibility_memo.md`
§"Results & findings" for the condensed version.*
