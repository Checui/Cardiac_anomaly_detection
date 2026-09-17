#!/usr/bin/env python
"""Flow-metric comparison on the baseline GAN: EPE / angular error vs flow-SSIM.

Scores the frozen baseline flow-GAN checkpoint
(ACDC_MM_RECON_ES_ED_FLOW_NOR_v2_orient_spacing_added, epoch 50) on held-out
ACDC-50 and M&Ms-Testing (plus M&Ms-Validation for honest selection), with the
anomaly score as the only thing varied:

  sanity anchors : flow_ssim, mag_ssim, flow_l1, appe (must reproduce the known
                   baseline flow-SSIM patient-mean AUC ACDC 0.8125 / MM 0.7310)
  standard flow  : flow_EPE (endpoint error), ang_barron (classic space-time
                   angular error of Barron et al. 1994)
  scale-free     : flow_EPEn, flow_ang (magnitude-weighted angular), flow_magr
                   (log magnitude ratio) -- the same definitions run on the QFAE
                   models via qfae_flowmetrics.extra_flow_scores, so the GAN cell
                   of that matrix becomes apples-to-apples.

Per dataset: patient-mean AUC per stream, patient-level bootstrap 95% CI, and a
paired bootstrap delta against flow_ssim (same resamples for both streams).
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
from utils import compute_flow_ssim_scores
from qfae_flowmetrics import extra_flow_scores, _EXTRA_FLOW_STREAMS
from sklearn.metrics import roc_auc_score

STREAMS = ('flow_ssim', 'mag_ssim', 'flow_l1', 'appe',
           'flow_EPE', 'flow_EPEn', 'flow_ang', 'flow_magr', 'ang_barron')
REF_STREAM = 'flow_ssim'


def barron_angular(gt, pred):
    """Classic space-time angular error (Barron et al. 1994), radians, mean over pixels.

    gt/pred are (H, W, 3) [dx, dy, mag]; only the displacement channels enter. Each flow
    vector is embedded as (u, v, 1) so the angle stays defined at zero motion.
    """
    u, v = gt[..., 0].astype(np.float64), gt[..., 1].astype(np.float64)
    uh, vh = pred[..., 0].astype(np.float64), pred[..., 1].astype(np.float64)
    num = 1.0 + u * uh + v * vh
    den = np.sqrt(1.0 + u ** 2 + v ** 2) * np.sqrt(1.0 + uh ** 2 + vh ** 2)
    return float(np.mean(np.arccos(np.clip(num / den, -1.0, 1.0))))


def score_pass(sess, t, images, flows, batch_size):
    """Frozen-BN scoring pass; returns per-sample score dict + predicted flows (float16)."""
    n = len(images)
    out = {k: np.zeros(n) for k in STREAMS}
    preds = np.zeros(flows.shape, dtype=np.float16)
    batches = np.array_split(np.arange(n), max(1, int(np.ceil(n / batch_size))))
    for vb in batches:
        appe, l1, pred_flow = sess.run(
            [t['ps_appe'], t['ps_opt'], t['out_opt']],
            feed_dict={
                t['frame']: images[vb],
                t['flow']:  flows[vb],
                t['is_training']: False,
            })
        fs, ms = compute_flow_ssim_scores(flows[vb], pred_flow)
        out['appe'][vb] = appe
        out['flow_l1'][vb] = l1
        out['flow_ssim'][vb] = fs
        out['mag_ssim'][vb] = ms
        for j, i in enumerate(vb):
            ex = extra_flow_scores(np.transpose(flows[i], (2, 0, 1)),
                                   np.transpose(pred_flow[j], (2, 0, 1)))
            for k in _EXTRA_FLOW_STREAMS:
                out[k][i] = ex[k]
            out['ang_barron'][i] = barron_angular(flows[i], pred_flow[j])
        preds[vb] = pred_flow.astype(np.float16)
    return out, preds


def patient_mean(scores, pids):
    by = {}
    for s, p in zip(scores, pids):
        by.setdefault(p, []).append(s)
    return {p: float(np.mean(v)) for p, v in by.items()}


def patient_auc(score_by_pid, label_by_pid):
    pids = sorted(score_by_pid)
    y = [0 if label_by_pid[p] == 'NOR' else 1 for p in pids]
    s = [score_by_pid[p] for p in pids]
    if len(set(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, s))


def bootstrap_stats(per_sample, pids, label_by_pid, n_boot=2000, seed=0):
    """Patient-level bootstrap: per-stream AUC CI + paired delta vs REF_STREAM.

    The same patient resample is used for every stream in a replicate, so the
    delta distribution is the paired one.
    """
    ps = sorted(set(pids))
    y = np.array([0 if label_by_pid[p] == 'NOR' else 1 for p in ps])
    mat = {}
    for k in STREAMS:
        pm = patient_mean(per_sample[k], pids)
        mat[k] = np.array([pm[p] for p in ps])
    rng = np.random.RandomState(seed)
    n = len(ps)
    aucs = {k: [] for k in STREAMS}
    deltas = {k: [] for k in STREAMS if k != REF_STREAM}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        ref = roc_auc_score(yy, mat[REF_STREAM][idx])
        aucs[REF_STREAM].append(ref)
        for k in deltas:
            a = roc_auc_score(yy, mat[k][idx])
            aucs[k].append(a)
            deltas[k].append(a - ref)
    stats = {}
    for k in STREAMS:
        a = np.array(aucs[k])
        stats[k] = {'ci': [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
        if k != REF_STREAM:
            d = np.array(deltas[k])
            stats[k]['delta_vs_%s' % REF_STREAM] = {
                'mean': float(d.mean()),
                'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                'frac_gt0': float((d > 0).mean()),
            }
    return stats


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
    ap.add_argument('--save_flows', action='store_true',
                    help='also save per-slice predicted + GT flows as float16 npz')
    ap.add_argument('--out_dir', default='./gan_flowmetric_out')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Loader config: the baseline checkpoint's recipe ───────────────────────
    # es_ed frame mode; orientation + spacing normalisation ON; N4 OFF;
    # Farneback aux (default backend).
    dl_flow.set_edes_direction('ed')
    dl_flow.set_orientation_normalization(True, args.orient_params)
    dl_flow.set_spacing_normalization(True, 1.5, 128, (2.0, 2.0))
    dl_flow.set_n4_bias_correction(False)

    # ── Held-out data ─────────────────────────────────────────────────────────
    print('=== Loading ACDC test set (full 50 patients) ===')
    (a_p1, a_p2, a_lbl, a_pid, a_slc,
     t_p1, t_p2, t_lbl, t_pid, t_slc) = dl_flow.load_acdc_test_val_ed_es_data(args.acdc_dir)
    if len(t_p1) > 0:
        a_p1 = np.concatenate([a_p1, t_p1], axis=0)
        a_p2 = np.concatenate([a_p2, t_p2], axis=0)
        a_lbl = list(a_lbl) + list(t_lbl)
        a_pid = list(a_pid) + list(t_pid)
    a_pid = ['ACDC_%s' % p for p in a_pid]
    print('ACDC: %d samples / %d patients' % (len(a_p1), len(set(a_pid))))

    print('=== Loading M&Ms Testing set ===')
    m_p1, m_p2, m_lbl, m_pid, m_slc = dl_flow.load_mm_validation_ed_es_data(
        args.mm_test_dir, args.mm_csv)
    m_pid = ['MM_%s' % p for p in m_pid]
    print('M&Ms-test: %d samples / %d patients' % (len(m_p1), len(set(m_pid))))

    print('=== Loading M&Ms Validation set (selection only) ===')
    v_p1, v_p2, v_lbl, v_pid, v_slc = dl_flow.load_mm_validation_ed_es_data(
        args.mm_val_dir, args.mm_csv)
    v_pid = ['MMVAL_%s' % p for p in v_pid]
    print('M&Ms-val: %d samples / %d patients' % (len(v_p1), len(set(v_pid))))

    datasets = {
        'ACDC':   (a_p1, a_p2, a_lbl, a_pid),
        'MM':     (m_p1, m_p2, m_lbl, m_pid),
        'MM_VAL': (v_p1, v_p2, v_lbl, v_pid),
    }
    label_by_pid = {}
    for _, (_, _, lbls, pids) in datasets.items():
        for p, l in zip(pids, lbls):
            prev = label_by_pid.setdefault(p, l)
            assert prev == l, 'inconsistent label for %s' % p

    # ── Graph: Generator + per-sample losses only (no D, no optimizer) ────────
    tf.reset_default_graph()
    h, w = a_p1.shape[1:3]
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

    tensors = {
        'frame': plh_frame_true, 'flow': plh_flow_true,
        'is_training': plh_is_training,
        'ps_appe': ps_loss_appe, 'ps_opt': ps_loss_opt, 'out_opt': output_opt,
    }

    ckpt = os.path.join(args.ckpt_dir, 'model_ckpt_%d.ckpt' % args.epoch)
    saver = tf.train.Saver(var_list=tf.global_variables())
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    results, pred_flows = {}, {}
    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt)
        print('Restored %s' % ckpt)
        for ds, (imgs, flows, _, _) in datasets.items():
            print('Scoring dataset=%s (%d samples) ...' % (ds, len(imgs)))
            results[ds], pred_flows[ds] = score_pass(
                sess, tensors, imgs, flows, batch_size=args.batch_size)

    # ── Patient-mean AUCs + bootstrap ─────────────────────────────────────────
    summary = {'checkpoint': ckpt, 'batch_size': args.batch_size,
               'n_boot': args.n_boot, 'ref_stream': REF_STREAM,
               'auc': {}, 'bootstrap': {}}
    pooled = {k: {} for k in STREAMS}
    for ds in datasets:
        _, _, _, pids = datasets[ds]
        summary['auc'][ds] = {}
        for k in STREAMS:
            pm = patient_mean(results[ds][k], pids)
            if ds != 'MM_VAL':
                pooled[k].update(pm)
            summary['auc'][ds][k] = patient_auc(pm, label_by_pid)
        print('Bootstrapping %s (%d reps) ...' % (ds, args.n_boot))
        summary['bootstrap'][ds] = bootstrap_stats(
            results[ds], pids, label_by_pid, n_boot=args.n_boot)
    summary['auc']['POOLED'] = {
        k: patient_auc(pooled[k], label_by_pid) for k in STREAMS}

    # Honest selection: the stream MM-Validation would pick, reported on held-out sets
    val_auc = summary['auc']['MM_VAL']
    sel = max(val_auc, key=lambda k: val_auc[k])
    summary['mmval_selected_stream'] = {
        'stream': sel, 'mmval_auc': val_auc[sel],
        'acdc_auc': summary['auc']['ACDC'][sel], 'mm_auc': summary['auc']['MM'][sel]}

    # ── Report ────────────────────────────────────────────────────────────────
    print('\n=== Patient-mean AUC (NOR vs disease) ===')
    print('%-12s' % 'stream'
          + ''.join('%22s' % ds for ds in ('ACDC', 'MM', 'MM_VAL', 'POOLED')))
    for k in STREAMS:
        row = '%-12s' % k
        for ds in ('ACDC', 'MM', 'MM_VAL', 'POOLED'):
            auc = summary['auc'][ds][k]
            if ds in summary['bootstrap']:
                ci = summary['bootstrap'][ds][k]['ci']
                row += '%9.4f [%.2f,%.2f]' % (auc, ci[0], ci[1])
            else:
                row += '%22.4f' % auc
        print(row)
    print('\n=== Paired bootstrap delta vs %s (mean [95%% CI], frac>0) ===' % REF_STREAM)
    for k in STREAMS:
        if k == REF_STREAM:
            continue
        row = '%-12s' % k
        for ds in ('ACDC', 'MM'):
            d = summary['bootstrap'][ds][k]['delta_vs_%s' % REF_STREAM]
            row += '   %+0.4f [%+0.3f,%+0.3f] %4.0f%%' % (
                d['mean'], d['ci'][0], d['ci'][1], 100 * d['frac_gt0'])
        print(row)
    print('\nMM-Validation would select: %s (val %.4f) -> held-out ACDC %.4f / MM %.4f'
          % (sel, val_auc[sel], summary['auc']['ACDC'][sel], summary['auc']['MM'][sel]))

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        os.path.join(args.out_dir, 'scores.npz'),
        **{'%s__%s' % (ds, k): results[ds][k] for ds in results for k in STREAMS},
        acdc_pids=np.array(a_pid), acdc_labels=np.array(a_lbl),
        mm_pids=np.array(m_pid), mm_labels=np.array(m_lbl),
        mmval_pids=np.array(v_pid), mmval_labels=np.array(v_lbl))
    if args.save_flows:
        np.savez_compressed(
            os.path.join(args.out_dir, 'flows.npz'),
            **{'pred__%s' % ds: pred_flows[ds] for ds in pred_flows},
            **{'gt__%s' % ds: datasets[ds][1].astype(np.float16) for ds in datasets})
        print('Saved flows.npz')
    print('\nSaved %s/summary.json and scores.npz' % args.out_dir)


if __name__ == '__main__':
    main()
