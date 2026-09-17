# Feasibility Memo — Pretrained CineMA encoder + reverse distillation for cardiac anomaly detection

## Context

The current cardiac pipeline (`Anomaly_detection_ICCV2019/`) trains an ICCV-2019 appearance–motion GAN **from scratch** on healthy (NOR) frames: a conv Inception encoder (`h0`–`h5` in `GAN_tf.py`), a reconstruction head, and an auxiliary motion head (optical flow / registration / next-frame). Anomalies are flagged as high reconstruction / flow error, scored per-patient via the existing aggregators (`Mean`, `FrameMax`, `SliceTop20`, …). The strongest current single-stream scorer is **Flow-SSIM** (patient-level `Mean` ≈ 0.73–0.77 AUC).

**Question evaluated:** instead of training an autoencoder from scratch, take a *pretrained* MAE, keep it frozen, and train only a smaller decoder / task head on healthy samples — with **CineMA** (cardiac-specific MAE) as the backbone and **reverse distillation** (feature-space) as the anomaly-detection paradigm. This memo assesses feasibility, surveys prior art, and flags the risks/open questions to resolve before committing. **No code changes are proposed here** — this is a decision document.

## Verdict

**Feasible and well-precedented.** "Frozen pretrained (M)AE encoder + small trainable head, learned on normal-only data" is an established 2023–2026 anomaly-detection recipe, and *feature-space* reverse distillation is currently the more robust form of it than pixel reconstruction. Two facts make the specific CineMA + RD instantiation attractive:

1. **CineMA is a released, cine-CMR-specific MAE** ([GitHub](https://github.com/mathpluscode/CineMA), [Hugging Face weights](https://huggingface.co/mathpluscode/CineMA), Nature *Communications Medicine* 2026) — a **convolutional-hybrid** MAE (ConvMAE-style conv downsampling ×8 → shared ViT), pretrained on 74,916 UK Biobank studies with 75% masking. It exposes **multi-scale dense feature maps** (used by its UNETR-style segmentation decoder), so a dense student decoder can tap it — this removes the ViT "no conv pyramid" mismatch that a plain ImageNet-MAE would have imposed.
2. Using a **domain-specific** teacher directly attacks the one limitation the RD literature repeatedly flags: with a *natural-image* frozen encoder, the domain gap starves the student of cardiac-specific cues. A cardiac teacher is the intended fix — and, to my knowledge, frozen-CineMA-as-RD-teacher for normal-only anomaly detection is **not yet published**, so it is a genuine (if incremental) contribution rather than a re-run.

## Proposed shape (conceptual, for discussion)

- **Freeze** the CineMA SAX encoder (teacher). Feed NOR cardiac frames; extract multi-scale features.
- Train a **small student decoder** (the only trainable part) to reproduce the teacher's features on **NOR-only** data — classic reverse distillation (Deng & Li, CVPR 2022).
- **Anomaly score** = multi-scale feature-reconstruction error, upsampled to a spatial anomaly map, then reduced to a per-sample scalar and aggregated to patient level with the **existing** aggregators in `GAN_tf.py` — so the whole val-AUC / per-disease / patient-level reporting harness is reusable and directly comparable to the current Flow-SSIM numbers.

## Why reverse distillation (feature-space) over pixel/flow reconstruction

Reconstruction AD in medical images suffers from **over-generalization / the "identity shortcut"**: a strong decoder reconstructs *anomalies* well too, collapsing the healthy-vs-disease error gap. Moving the reconstruction target from pixels to **frozen teacher features** (reverse distillation, PatchCore/PaDiM density on features) is the field's response and is more robust with a frozen encoder. This matches the user's own instinct ("train a small decoder on healthy features").

## Annotated related work

| Work | Relevance |
|---|---|
| **CineMA** — Foundation model for cine CMR ([2506.00679](https://arxiv.org/abs/2506.00679), [code](https://github.com/mathpluscode/CineMA), [HF](https://huggingface.co/mathpluscode/CineMA)) | The proposed backbone. Conv-hybrid MAE, 75% masking, multi-scale dense features. **Caveat:** its headline **ACDC disease AUC 97.98%** came from *full fine-tuning with disease labels* — not a frozen encoder and not normal-only, so it is an *upper bound under a different regime*, not evidence a frozen probe will hit ~0.98. |
| **AMAE** ([2307.12721](https://arxiv.org/abs/2307.12721)) | Almost the exact proposal: pretrained MAE, frozen features, synthetic anomalies from normal-only images, lightweight head. Chest X-ray. |
| **Q-Former Autoencoder** (WACV 2026, [paper](https://openaccess.thecvf.com/content/WACV2026/papers/Dalmonte_Q-Former_Autoencoder_A_Modern_Framework_for_Medical_Anomaly_Detection_WACV_2026_paper.pdf)) | Frozen foundation models (DINOv2, MAE) as feature extractors + small decoder for medical AD. Validates "frozen pretrained + small trainable head." |
| **MAE-medical** — Georgescu et al. ([2307.07534](https://arxiv.org/pdf/2307.07534), [code](https://github.com/lilygeorgescu/MAE-medical-anomaly-detection)) | MAE learns normal structure; anomaly score from reconstruction difference. Pixel-space baseline to compare against. |
| **Reverse Distillation** family — Deng & Li CVPR 2022; DNP-RD ([2508.19573](https://arxiv.org/abs/2508.19573)) | The chosen paradigm. DNP-RD explicitly notes frozen general encoders "struggle to extract domain-specific features" when the domain gap is large — motivating the CineMA teacher. |
| **MemMC-MAE** ([2203.11725](https://arxiv.org/pdf/2203.11725)) | Memory-augmented MAE for medical AD; relevant if over-generalization needs an extra brake. |
| **Brain UAD benchmark** ([2512.01534](https://arxiv.org/pdf/2512.01534)) | Large-scale benchmarking/bias analysis of unsupervised AD in medical imaging — useful for framing evaluation honestly. |

## On "would VideoMAE make sense?"

The *temporal masked-AE idea* is right for cine data, but **VideoMAE is pretrained on natural video (Kinetics)** whose temporal statistics (camera motion, scene cuts) are unlike cyclic cardiac contraction — a large domain gap. **A cardiac-domain MAE (CineMA) is the better choice than VideoMAE** for this modality. Note CineMA's SAX pathway already ingests a spatiotemporal stack (192×192×16), so temporal information can live *inside* the frozen features without needing VideoMAE.

## CineMA fit — specifics and mismatches to resolve

- **Architecture (good):** conv-hybrid + multi-scale dense features → compatible with a UNETR/RD-style student. Better fit than a plain ViT MAE.
- **Weights/license:** released on GitHub + Hugging Face; **confirm the license permits your research use** before building on it.
- **Input geometry (adaptation cost):** CineMA is **multi-view** (LAX 2C/3C/4C at 256², SAX at 192²×16), each view encoded independently. Your ACDC/M&Ms pipeline is **SAX-only, single 2D frames letterboxed to 128²**. You would use only CineMA's SAX branch and must reconcile resolution (128→192) and whether to feed single frames or the 16-frame stack. Verify the SAX encoder runs stand-alone.
- **Frozen ≠ their 98%:** CineMA's strong disease number is *supervised full fine-tune*. Frozen + normal-only RD is unproven for CineMA — that gap is exactly your experiment.

## Risks / open questions

1. **Domain shift within cardiac:** CineMA pretrained on UK Biobank; ACDC/M&Ms differ in scanner/protocol/pathology mix. Frozen features may still under-represent subtle wall-motion pathology (HCM/DCM) — the very cases your appearance stream already struggles with.
2. **Where does motion go?** Classic RD is spatial-only and would **drop the explicit flow head** — yet Flow-SSIM is currently your *best* scorer. Options: feed CineMA the temporal SAX stack so motion is embedded in features, or keep a lightweight flow head alongside RD as a complementary stream. Decide before discarding the motion signal.
3. **Loses interpretability:** the GAN gives an interpretable predicted flow field; RD gives a feature-error heatmap. Acceptable, but note it for the write-up.
4. **Frozen-encoder ceiling:** if frozen CineMA underperforms, a light adapter/LoRA fine-tune of the encoder on NOR frames is the natural fallback (still normal-only).

## Suggested minimal de-risking experiment (before any commitment)

A one-afternoon check that needs no changes to the GAN: load frozen CineMA (HF weights), run the **existing NOR + disease val frames** through its SAX encoder, and test whether a simple normal-feature model already separates disease — e.g. **PatchCore/Mahalanobis distance on the frozen features** (no student training required). Score with the current patient-level aggregators and compare AUC to the Flow-SSIM baseline (~0.73–0.77). If frozen CineMA features separate disease at all, the full reverse-distillation build is justified; if they don't, that's a strong early signal to reconsider the backbone or add encoder adaptation.

## Implementation & where it runs — **HPC always**

> **All CineMA / MAE experiments for this project run on the Imperial College HPC (PBS), not locally.** Datasets, cached model weights, and the conda env all live on the HPC. Submit with `qsub`; the login node is contended, so do not run training/feature extraction there (short read-only checks only).

The de-risking experiment above is **implemented** (built out from the "one-afternoon check"):

- **Submit:** `qsub derisk_cinema.pbs` from the repo root → batch job on a GPU compute node → results in `derisk_out/derisk_results.json` (+ `derisk_arrays.npz`, and the PBS `Derisk_CineMA.o<jobid>` / `.e<jobid>` logs).
- **Code:** `derisk_cinema.py` (entry point; `--cinema_preproc faithful|legacy`, Mahalanobis + kNN on frozen features), `cinema_faithful.py` (CineMA-canonical SAX preprocessing: resample 1 mm → LV-bbox crop 192² → clip 0.95/99.5 → real depth-16 stacks), `derisk_cinema.pbs` (submission script).
- **Env:** conda env `derisk` — torch + editable `cinema` (from `~/CineMA`) + monai / scikit-learn / scikit-image / SimpleITK / nibabel / opencv. CineMA weights are cached **offline** at `$EPHEMERAL/hf_cache` (the PBS job runs with `HF_HUB_OFFLINE=1`); pre-download once on the login node if the cache is ever cleared.
- **Faithful vs legacy:** `faithful` (default) reads raw ACDC/M&Ms via CineMA's own pipeline (on-distribution); `legacy` uses the ICCV `data_loader.py` single-2D-slice path (honours `--orient/--spacing/--n4`). The faithful path scores per 3-D stack (one per patient×phase) and aggregates to patient by Mean/Max.

## Results & findings (experiments, 2026-07)

The de-risk was built out and run on the HPC (frozen-feature probe: Mahalanobis /
Ledoit-Wolf + kNN on NOR-only CineMA features, scored NOR-vs-disease, patient-level).
All AUCs below are **patient-level Mean, Mahalanobis** (the strongest scorer; kNN was
uniformly weaker).

| Configuration | Overall | ACDC | M&Ms |
|---|---|---|---|
| Frozen CineMA, **faithful** canonical-SAX preprocessing | 0.68 | **0.76** | 0.59 |
| Frozen CineMA, legacy single-2D-slice preprocessing | 0.59 | 0.62 | 0.51 |
| **NOR-only LoRA adapter** (continued MAE pretrain, best of 2 configs) | 0.68 | **0.79** | 0.58 |
| Flow-SSIM motion GAN baseline (current pipeline) | **0.73–0.77** | — | — |

**Findings:**
1. **Frozen CineMA carries real in-distribution disease signal.** On ACDC it reaches
   0.76 with no decoder training — matching the Flow-SSIM baseline — confirming the memo's
   central hypothesis for single-centre data.
2. **Reading it on-distribution is essential.** CineMA's own canonical SAX pipeline
   (resample 1 mm → LV-bbox crop 192² → clip → real depth-16 stacks) beats a crude
   single-2D-slice feed by **+0.145 AUC on ACDC** (0.76 vs 0.62), even though the crude
   path had ~3× more samples. Input geometry, not data volume, was the lever.
3. **The M&Ms multi-vendor domain gap is genuine and robust.** M&Ms sits at ~0.59 and is
   *not* a preprocessing bug (LV labels/crops verified). It is closed by **neither**
   feature-config/PCA tuning **nor** light NOR-only MAE adaptation. A LoRA adapter across
   **three** configurations — minimal; constrained-decoder + stronger LoRA; and **all-cine-
   frames (~13× the NOR data)** — consistently *sharpened ACDC to ~0.79* (above baseline)
   while M&Ms stayed flat (ACDC/M&Ms ΔAUC: +0.006/+0.003, +0.032/−0.009, +0.032/−0.000).
   So the gap is not a decoder-capacity, learning-rate, or **data-volume** problem: light
   self-supervised normal-only adaptation reliably helps the single-centre set and simply
   does not transfer to the multi-vendor one.
4. **Overall, the frozen/adapted CineMA probe (~0.68) does not beat the Flow-SSIM GAN
   (0.73–0.77) on the combined ACDC+M&Ms benchmark** — the M&Ms gap holds it back. CineMA
   wins in-distribution (ACDC) but not across domains.

**Conclusion.** CineMA is a strong *in-distribution* cardiac anomaly detector as a frozen,
normal-only feature extractor, and the canonical-SAX pipeline is what unlocks it. But in
this label-free regime it does not surpass the existing motion-based GAN on the
multi-vendor benchmark, because light self-supervised adaptation cannot cross the M&Ms
domain gap. Closing that gap would require either **more adaptation data** (all cardiac
phases, ~10× more NOR stacks — untried) or **stronger/supervised adaptation** (leaving the
strict normal-only regime). Artefacts on the HPC: `derisk_out/` (frozen),
`derisk_out_legacy/`, `derisk_out_adapted_v2/` (adapted); adapter in `adapter_out_v2/`.

## Sources

- CineMA: https://arxiv.org/abs/2506.00679 · https://github.com/mathpluscode/CineMA · https://huggingface.co/mathpluscode/CineMA · https://www.nature.com/articles/s43856-026-01636-0
- AMAE: https://arxiv.org/abs/2307.12721
- Q-Former Autoencoder (WACV 2026): https://openaccess.thecvf.com/content/WACV2026/papers/Dalmonte_Q-Former_Autoencoder_A_Modern_Framework_for_Medical_Anomaly_Detection_WACV_2026_paper.pdf
- MAE-medical (Georgescu et al.): https://arxiv.org/pdf/2307.07534 · https://github.com/lilygeorgescu/MAE-medical-anomaly-detection
- Reverse Distillation / DNP-RD: https://arxiv.org/abs/2508.19573
- MemMC-MAE: https://arxiv.org/pdf/2203.11725
- Brain UAD benchmark: https://arxiv.org/pdf/2512.01534
