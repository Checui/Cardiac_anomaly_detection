#!/usr/bin/env python
"""Reverting the cardiac adaptations on the baseline GAN, one at a time.

Scores the frozen baseline flow-GAN checkpoint
(ACDC_MM_RECON_ES_ED_FLOW_NOR_v2_orient_spacing_added, epoch 50) on held-out
ACDC-50 and M&Ms-Testing (plus M&Ms-Validation for the honest-selection
column), with each adaptation of the thesis' Chapter 4 undone in turn:

  headline          flow-SSIM, full frame, middle-60%% slices, patient mean
                    (sanity anchor: must reproduce ACDC 0.8125 / MM 0.7310)
  revert R restr.   same score, patient mean over ALL slices (the middle-60%%
                    crop in data_loader is monkeypatched off for a 2nd pass;
                    scoring-side revert only -- the checkpoint stays trained
                    on middle-60%% data)
  revert support    the paper's 16x16 max-flow-error patch (utils.
                    compute_patch_scores) instead of full-frame pooling
  revert score      paper eq. 8, log(S_F/mu_F) + 0.2 log(S_I/mu_I) on the
                    patch, vs the adapted full-frame 2:1 combination; both
                    use the training-set mu from mu_baseline_<epoch>.json
  revert unit       slice-level AUC (patient label broadcast to slices)
                    instead of patient-level, with and without the per-video
                    max-normalisation analogue (per-patient max)
  original protocol all reverts together: patch eq-8 score, all slices,
                    per-patient max normalisation, slice-level AUC

Outputs gan_revert_out/{summary.json, scores.npz}; per-sample scores are kept
with pids and slice indices for both passes so the table can be re-derived
without another GPU pass.
"""
import sys
import os
import json
import argparse
import types

# Mock ProgressBar and alias TF v1 exactly as run_model.py does
class MockProgressBar:
    FULL = 'full'
    def __init__(self, n, fmt='full'):
        self.n = n
        self.current = 0
    def __call__(self): pass
    def done(self): pass

mock_pb_module = types.ModuleType("ProgressBar")
mock_pb_module.ProgressBar = MockProgressBar
sys.modules["ProgressBar"] = mock_pb_module

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.modules['tensorflow'] = tf

import numpy as np
import GAN_tf
import data_loader as dl_flow
from utils import compute_flow_ssim_scores, compute_patch_scores
from sklearn.metrics import roc_auc_score

EPS = 1e-10
BASE_STREAMS = ('appe', 'flow_l1', 'flow_ssim', 'mag_ssim', 'patch_flow', 'patch_appe')
DERIVED_STREAMS = ('comb_ff', 'comb_patch')
STREAMS = BASE_STREAMS + DERIVED_STREAMS

# The rows of tab:gan_revert: (name, pass, stream, aggregation)
CONFIGS = [
    ('headline_flow_ssim',     'mid60', 'flow_ssim',  'patient_mean'),
    ('anchor_appe',            'mid60', 'appe',       'patient_mean'),
    ('revert_all_slices',      'all',   'flow_ssim',  'patient_mean'),
    ('revert_patch_flow',      'mid60', 'patch_flow', 'patient_mean'),
    ('revert_patch_appe',      'mid60', 'patch_appe', 'patient_mean'),
    ('adapted_comb_fullframe', 'mid60', 'comb_ff',    'patient_mean'),
    ('revert_comb_patch_eq8',  'mid60', 'comb_patch', 'patient_mean'),
    ('revert_slice_level',     'mid60', 'flow_ssim',  'slice'),
    ('revert_slice_pmax',      'mid60', 'flow_ssim',  'slice_pmax'),
    ('original_protocol',      'all',   'comb_patch', 'slice_pmax'),
]
REF_CONFIG = 'headline_flow_ssim'


def score_pass(sess, t, images, flows, batch_size):
    """Frozen-BN scoring pass; returns dict of per-sample base streams."""
    n = len(images)
    out = {k: np.zeros(n) for k in BASE_STREAMS}
    batches = np.array_split(np.arange(n), max(1, int(np.ceil(n / batch_size))))
    for vb in batches:
        appe, l1, raw_f, raw_a, pred_flow = sess.run(
            [t['ps_appe'], t['ps_opt'], t['raw_flow'], t['raw_appe'], t['out_opt']],
            feed_dict={
                t['frame']: images[vb],
                t['flow']:  flows[vb],
                t['is_training']: False,
            })
        out['appe'][vb] = appe
        out['flow_l1'][vb] = l1
        pf, pa = compute_patch_scores(raw_f, raw_a)
        out['patch_flow'][vb] = pf
        out['patch_appe'][vb] = pa
        fs, ms = compute_flow_ssim_scores(flows[vb], pred_flow)
        out['flow_ssim'][vb] = fs
        out['mag_ssim'][vb] = ms
    return out


def add_derived(out, mu):
    """Combined scores from the training-set mu baselines (mu_baseline_<ep>.json).

    comb_ff    -- adapted full-frame 2:1 combination (GAN_tf.py validation loop)
    comb_patch -- paper eq. 8, lambda_S = 0.2, on the max-flow-error patch
    """
    out['comb_ff'] = (
        np.log(np.maximum(out['appe'], EPS) / max(mu['mu_appe'], EPS))
        + 2.0 * np.log(np.maximum(out['flow_l1'], EPS) / max(mu['mu_aux'], EPS)))
    out['comb_patch'] = (
        np.log(np.maximum(out['patch_flow'], EPS) / max(mu['mu_aux_patch'], EPS))
        + 0.2 * np.log(np.maximum(out['patch_appe'], EPS) / max(mu['mu_appe_patch'], EPS)))
    return out


def patient_mean(scores, pids):
    by = {}
    for s, p in zip(scores, pids):
        by.setdefault(p, []).append(s)
    return {p: float(np.mean(v)) for p, v in by.items()}


def patient_max_normalize(scores, pids):
    """Paper eq. 11 analogue: divide each sample score by its patient's max.

    Guarded: a patient whose max is <= 0 (possible for the log-scale combined
    scores) is left unnormalised; the guard count is returned so the caveat can
    be reported.
    """
    s = np.asarray(scores, dtype=float).copy()
    pids = np.asarray(pids)
    guarded = 0
    for p in np.unique(pids):
        m = pids == p
        mx = s[m].max()
        if mx > 0:
            s[m] = s[m] / mx
        else:
            guarded += 1
    return s, guarded


def patient_auc(score_by_pid, label_by_pid):
    pids = sorted(score_by_pid)
    y = [0 if label_by_pid[p] == 'NOR' else 1 for p in pids]
    s = [score_by_pid[p] for p in pids]
    if len(set(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, s))


def slice_auc(scores, pids, label_by_pid):
    y = [0 if label_by_pid[p] == 'NOR' else 1 for p in pids]
    if len(set(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, scores))


def build_config_repr(results, pids_by_pass, label_by_pid):
    """Per-config, per-patient representation for AUC + cluster bootstrap.

    patient_mean -> {pid: scalar}; slice / slice_pmax -> {pid: np.array of the
    patient's per-sample scores (pmax already applied for slice_pmax)}.
    Returns (reprs, guard_counts).
    """
    reprs, guards = {}, {}
    for name, pas, stream, agg in CONFIGS:
        scores = results[pas][stream]
        pids = pids_by_pass[pas]
        if agg == 'patient_mean':
            reprs[name] = ('scalar', patient_mean(scores, pids))
            continue
        if agg == 'slice_pmax':
            scores, g = patient_max_normalize(scores, pids)
            guards[name] = g
        by = {}
        for s, p in zip(scores, pids):
            by.setdefault(p, []).append(s)
        reprs[name] = ('samples', {p: np.array(v) for p, v in by.items()})
    return reprs, guards


def config_point_aucs(reprs, label_by_pid):
    aucs = {}
    for name, (kind, by) in reprs.items():
        if kind == 'scalar':
            aucs[name] = patient_auc(by, label_by_pid)
        else:
            s = np.concatenate([by[p] for p in sorted(by)])
            p_all = np.concatenate([[p] * len(by[p]) for p in sorted(by)])
            aucs[name] = slice_auc(s, p_all, label_by_pid)
    return aucs


def cluster_bootstrap(reprs, label_by_pid, n_boot=2000, seed=0):
    """Bootstrap over patients (the sampling unit for every config, including
    the slice-level ones -- slices of one patient are not independent). The
    same patient resample feeds every config, so deltas vs REF_CONFIG are
    paired."""
    ps = sorted(set.intersection(*[set(by) for _, by in reprs.values()]))
    y = np.array([0 if label_by_pid[p] == 'NOR' else 1 for p in ps])
    rng = np.random.RandomState(seed)
    n = len(ps)
    aucs = {k: [] for k in reprs}
    deltas = {k: [] for k in reprs if k != REF_CONFIG}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        rep_aucs = {}
        for name, (kind, by) in reprs.items():
            if kind == 'scalar':
                s = np.array([by[ps[i]] for i in idx])
                yv = yy
            else:
                s = np.concatenate([by[ps[i]] for i in idx])
                yv = np.concatenate([[yy[j]] * len(by[ps[i]]) for j, i in enumerate(idx)])
            rep_aucs[name] = roc_auc_score(yv, s)
        ref = rep_aucs[REF_CONFIG]
        for name, a in rep_aucs.items():
            aucs[name].append(a)
            if name != REF_CONFIG:
                deltas[name].append(a - ref)
    if not aucs[REF_CONFIG]:  # every resample was single-class
        nanci = {'ci': [float('nan'), float('nan')]}
        return {name: dict(nanci, **({} if name == REF_CONFIG else {
            'delta_vs_headline': {'mean': float('nan'),
                                  'ci': [float('nan'), float('nan')],
                                  'frac_gt0': float('nan')}}))
                for name in reprs}
    stats = {}
    for name in reprs:
        a = np.array(aucs[name])
        stats[name] = {'ci': [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
        if name != REF_CONFIG:
            d = np.array(deltas[name])
            stats[name]['delta_vs_headline'] = {
                'mean': float(d.mean()),
                'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                'frac_gt0': float((d > 0).mean()),
            }
    return stats


def load_datasets(args, smoke=False):
    """One load of every evaluation set under the CURRENT loader state.

    Returns {ds: (images, flows, labels, pids, slice_idxs)}.
    """
    datasets = {}
    if not smoke:
        print('=== Loading ACDC test set (full 50 patients) ===')
        (a_p1, a_p2, a_lbl, a_pid, a_slc,
         t_p1, t_p2, t_lbl, t_pid, t_slc) = dl_flow.load_acdc_test_val_ed_es_data(args.acdc_dir)
        if len(t_p1) > 0:
            a_p1 = np.concatenate([a_p1, t_p1], axis=0)
            a_p2 = np.concatenate([a_p2, t_p2], axis=0)
            a_lbl = list(a_lbl) + list(t_lbl)
            a_pid = list(a_pid) + list(t_pid)
            a_slc = list(a_slc) + list(t_slc)
        a_pid = ['ACDC_%s' % p for p in a_pid]
        datasets['ACDC'] = (a_p1, a_p2, list(a_lbl), a_pid, list(a_slc))

        print('=== Loading M&Ms Testing set ===')
        m_p1, m_p2, m_lbl, m_pid, m_slc = dl_flow.load_mm_validation_ed_es_data(
            args.mm_test_dir, args.mm_csv)
        m_pid = ['MM_%s' % p for p in m_pid]
        datasets['MM'] = (m_p1, m_p2, list(m_lbl), m_pid, list(m_slc))

    print('=== Loading M&Ms Validation set (selection only) ===')
    v_p1, v_p2, v_lbl, v_pid, v_slc = dl_flow.load_mm_validation_ed_es_data(
        args.mm_val_dir, args.mm_csv)
    v_pid = ['MMVAL_%s' % p for p in v_pid]
    datasets['MM_VAL'] = (v_p1, v_p2, list(v_lbl), v_pid, list(v_slc))

    if smoke:
        n = args.smoke_n
        ds = datasets['MM_VAL']
        datasets['MM_VAL'] = tuple(x[:n] for x in ds)
    for ds, (imgs, _, _, pids, _) in datasets.items():
        print('%s: %d samples / %d patients' % (ds, len(imgs), len(set(pids))))
    return datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', default='./training_saver/ACDC_MM_RECON_ES_ED_FLOW_NOR_v2_orient_spacing_added')
    ap.add_argument('--epoch', type=int, default=50)
    ap.add_argument('--acdc_dir', default='../Dataset_2')
    ap.add_argument('--mm_test_dir', default='../Dataset_1/Testing')
    ap.add_argument('--mm_val_dir', default='../Dataset_1/Validation')
    ap.add_argument('--mm_csv', default='../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
    ap.add_argument('--orient_params', default='../reconstructed_sax_images_training_2023/segmentation/orientation_params.csv')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--out_dir', default='./gan_revert_out')
    ap.add_argument('--smoke', action='store_true',
                    help='M&Ms Validation only, truncated, 50 bootstrap reps; '
                         'end-to-end sanity check before qsub')
    ap.add_argument('--smoke_n', type=int, default=120)
    args = ap.parse_args()
    if args.smoke:
        args.n_boot = 50

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Loader config: the baseline checkpoint's recipe ───────────────────────
    # es_ed frame mode; orientation + spacing normalisation ON; N4 OFF;
    # Farneback aux (default backend).
    dl_flow.set_edes_direction('ed')
    dl_flow.set_orientation_normalization(True, args.orient_params)
    dl_flow.set_spacing_normalization(True, 1.5, 128, (2.0, 2.0))
    dl_flow.set_n4_bias_correction(False)

    # ── Training-set mu baselines (saved by the training run) ─────────────────
    mu_path = os.path.join(args.ckpt_dir, 'mu_baseline_%d.json' % args.epoch)
    with open(mu_path) as f:
        mu = json.load(f)
    print('mu baselines: %s' % {k: v for k, v in mu.items() if k.startswith('mu_')})

    # ── Graph: Generator + per-sample losses + raw diff maps (no D) ───────────
    tf.reset_default_graph()
    h = w = 128
    plh_frame_true = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    plh_flow_true = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    plh_is_training = tf.placeholder(tf.bool)
    scaled_frame_true = (plh_frame_true / 0.5) - 1.0
    plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
    output_opt, output_appe = GAN_tf.Generator(scaled_frame_true, plh_is_training, plh_dropout_prob)

    dy1, dx1 = tf.image.image_gradients(output_appe)
    dy0, dx0 = tf.image.image_gradients(scaled_frame_true)
    ps_loss_inten = tf.reduce_mean((output_appe - scaled_frame_true) ** 2, axis=[1, 2, 3])
    ps_loss_gradi = tf.reduce_mean(
        tf.abs(tf.abs(dy1) - tf.abs(dy0)) + tf.abs(tf.abs(dx1) - tf.abs(dx0)),
        axis=[1, 2, 3])
    ps_loss_appe = ps_loss_inten + ps_loss_gradi
    ps_loss_opt = tf.reduce_mean(tf.abs(output_opt - plh_flow_true), axis=[1, 2, 3])
    # Raw 2D diff maps for patch scoring (reduce over channels only), as in
    # GAN_tf.py's validation loop
    raw_diff_map_flow = tf.reduce_mean((output_opt - plh_flow_true) ** 2, axis=-1)
    raw_diff_map_appe = tf.reduce_mean((output_appe - scaled_frame_true) ** 2, axis=-1)

    tensors = {
        'frame': plh_frame_true, 'flow': plh_flow_true,
        'is_training': plh_is_training,
        'ps_appe': ps_loss_appe, 'ps_opt': ps_loss_opt, 'out_opt': output_opt,
        'raw_flow': raw_diff_map_flow, 'raw_appe': raw_diff_map_appe,
    }

    ckpt = os.path.join(args.ckpt_dir, 'model_ckpt_%d.ckpt' % args.epoch)
    saver = tf.train.Saver(var_list=tf.global_variables())
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # results[pass][ds][stream] -> per-sample array; meta[pass][ds] -> (lbl, pid, slc)
    results = {'mid60': {}, 'all': {}}
    meta = {'mid60': {}, 'all': {}}
    label_by_pid = {}

    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt)
        print('Restored %s' % ckpt)

        for pas in ('mid60', 'all'):
            if pas == 'all':
                # Revert the slice restriction: every call site resolves
                # _middle_slice_range through the module namespace, so
                # replacing it keeps all slices without touching the repo.
                dl_flow._middle_slice_range = lambda Z, frac=0.2: range(Z)
                print('\n### Pass 2: middle-60%% restriction DISABLED (all slices) ###')
            else:
                print('\n### Pass 1: default loaders (middle-60%% slices) ###')
            datasets = load_datasets(args, smoke=args.smoke)
            for ds, (imgs, flows, lbls, pids, slcs) in datasets.items():
                for p, l in zip(pids, lbls):
                    prev = label_by_pid.setdefault(p, l)
                    assert prev == l, 'inconsistent label for %s' % p
                print('Scoring %s / %s (%d samples) ...' % (pas, ds, len(imgs)))
                results[pas][ds] = add_derived(
                    score_pass(sess, tensors, imgs, flows, batch_size=args.batch_size), mu)
                meta[pas][ds] = (list(lbls), list(pids), list(slcs))
            del datasets

    # ── Table: point AUCs + patient-cluster bootstrap per dataset ─────────────
    ds_names = [d for d in ('ACDC', 'MM', 'MM_VAL') if d in results['mid60']]
    summary = {'checkpoint': ckpt, 'mu': mu, 'batch_size': args.batch_size,
               'n_boot': args.n_boot, 'ref_config': REF_CONFIG,
               'configs': [list(c) for c in CONFIGS],
               'n_samples': {pas: {ds: len(meta[pas][ds][1]) for ds in results[pas]}
                             for pas in results},
               'auc': {}, 'bootstrap': {}, 'pmax_guard': {}}
    for ds in ds_names:
        pids_by_pass = {pas: meta[pas][ds][1] for pas in ('mid60', 'all')}
        p_mid, p_all = set(pids_by_pass['mid60']), set(pids_by_pass['all'])
        if p_mid != p_all:
            print('WARNING %s: patient sets differ between passes '
                  '(mid60 %d vs all %d); bootstrap uses the intersection'
                  % (ds, len(p_mid), len(p_all)))
        per_ds_results = {pas: results[pas][ds] for pas in ('mid60', 'all')}
        reprs, guards = build_config_repr(per_ds_results, pids_by_pass, label_by_pid)
        summary['auc'][ds] = config_point_aucs(reprs, label_by_pid)
        summary['pmax_guard'][ds] = guards
        if ds != 'MM_VAL' or args.smoke:
            print('Bootstrapping %s (%d reps) ...' % (ds, args.n_boot))
            summary['bootstrap'][ds] = cluster_bootstrap(
                reprs, label_by_pid, n_boot=args.n_boot)

    # ── Report ────────────────────────────────────────────────────────────────
    print('\n=== Reverting the adaptations: AUC (NOR vs disease) ===')
    print('%-24s' % 'config' + ''.join('%22s' % ds for ds in ds_names))
    for name, pas, stream, agg in CONFIGS:
        row = '%-24s' % name
        for ds in ds_names:
            auc = summary['auc'][ds][name]
            if ds in summary['bootstrap']:
                ci = summary['bootstrap'][ds][name]['ci']
                row += '%9.4f [%.2f,%.2f]' % (auc, ci[0], ci[1])
            else:
                row += '%22.4f' % auc
        print(row)
    print('\n=== Paired bootstrap delta vs %s (mean [95%% CI], frac>0) ===' % REF_CONFIG)
    for name, _, _, _ in CONFIGS:
        if name == REF_CONFIG:
            continue
        row = '%-24s' % name
        for ds in ds_names:
            if ds not in summary['bootstrap']:
                continue
            d = summary['bootstrap'][ds][name]['delta_vs_headline']
            row += '   %+0.4f [%+0.3f,%+0.3f] %4.0f%%' % (
                d['mean'], d['ci'][0], d['ci'][1], 100 * d['frac_gt0'])
        print(row)
    for ds in ds_names:
        g = {k: v for k, v in summary['pmax_guard'][ds].items() if v}
        if g:
            print('pmax guard (%s): %s patients left unnormalised (max <= 0)' % (ds, g))

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        os.path.join(args.out_dir, 'scores.npz'),
        **{'%s__%s__%s' % (pas, ds, k): results[pas][ds][k]
           for pas in results for ds in results[pas] for k in STREAMS},
        **{'%s__%s__pids' % (pas, ds): np.array(meta[pas][ds][1])
           for pas in meta for ds in meta[pas]},
        **{'%s__%s__labels' % (pas, ds): np.array(meta[pas][ds][0])
           for pas in meta for ds in meta[pas]},
        **{'%s__%s__slice_idx' % (pas, ds): np.array(meta[pas][ds][2])
           for pas in meta for ds in meta[pas]})
    print('\nSaved %s/summary.json and scores.npz' % args.out_dir)


if __name__ == '__main__':
    main()
