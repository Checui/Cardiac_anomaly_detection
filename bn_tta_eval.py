#!/usr/bin/env python
"""Test-time BatchNorm adaptation (TTBN) falsification test.

Scores the frozen baseline flow-GAN checkpoint on the held-out validation sets
twice, changing ONLY the BN statistics source:

  frozen    : plh_is_training=False -- moving-average statistics (normal eval;
              must reproduce the known flow-SSIM patient-mean baseline
              ACDC 0.8125 / M&Ms-test 0.731).
  ttbn      : plh_is_training=True  -- per-batch statistics, sequential batches
              (data is ordered patient-by-patient, so batches are nearly
              single-patient = per-patient adaptation).
  ttbn_shuf : plh_is_training=True, batches drawn from a seeded shuffle within
              each dataset (mixed-patient / mixed-vendor batches).

Rationale: the "IN in the encoder" proposal rests on frozen train-vendor BN
statistics mis-normalising unseen-vendor activations. Test-time BN adaptation
(Nado et al.) is the direct intervention on that mechanism with zero retraining:
if the M&Ms AUC does not move under TTBN, the BN-mismatch hypothesis is dead for
this model and the IN ablation is moot. Dropout is controlled by a separate
placeholder (keep=1.0 throughout), so flipping is_training changes BN only.
No update ops are fetched, so moving averages / the checkpoint are untouched.

Datasets are scored separately so no batch ever spans ACDC and M&Ms.
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
from sklearn.metrics import roc_auc_score

STREAMS = ('flow_ssim', 'mag_ssim', 'flow_l1', 'appe')


def score_pass(sess, t, images, flows, is_training, batch_size, shuffle_seed=None):
    """One full scoring pass; returns per-sample score arrays (original order)."""
    n = len(images)
    order = np.arange(n)
    if shuffle_seed is not None:
        order = np.random.RandomState(shuffle_seed).permutation(n)
    out = {k: np.zeros(n) for k in STREAMS}
    batches = np.array_split(order, max(1, int(np.ceil(n / batch_size))))
    for vb in batches:
        appe, l1, pred_flow = sess.run(
            [t['ps_appe'], t['ps_opt'], t['out_opt']],
            feed_dict={
                t['frame']: images[vb],
                t['flow']:  flows[vb],
                t['is_training']: is_training,
            })
        fs, ms = compute_flow_ssim_scores(flows[vb], pred_flow)
        out['appe'][vb] = appe
        out['flow_l1'][vb] = l1
        out['flow_ssim'][vb] = fs
        out['mag_ssim'][vb] = ms
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', default='./training_saver/ACDC_MM_RECON_ES_ED_FLOW_NOR_v2_orient_spacing_added')
    ap.add_argument('--epoch', type=int, default=50)
    ap.add_argument('--acdc_dir', default='../Dataset_2')
    ap.add_argument('--mm_test_dir', default='../Dataset_1/Testing')
    ap.add_argument('--mm_csv', default='../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
    ap.add_argument('--orient_params', default='../reconstructed_sax_images_training_2023/segmentation/orientation_params.csv')
    ap.add_argument('--batch_size', type=int, default=16,
                    help='16 = the training batch size, so TTBN statistics have the same noise level BN was trained with.')
    ap.add_argument('--out_dir', default='./bn_tta_out')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Loader config: the baseline checkpoint's recipe ───────────────────────
    # es_ed frame mode; orientation + spacing normalisation ON; N4 OFF;
    # Farneback aux (default backend).
    dl_flow.set_edes_direction('ed')
    dl_flow.set_orientation_normalization(True, args.orient_params)
    dl_flow.set_spacing_normalization(True, 1.5, 128, (2.0, 2.0))
    dl_flow.set_n4_bias_correction(False)

    # ── Held-out validation data ──────────────────────────────────────────────
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

    datasets = {
        'ACDC': (a_p1, a_p2, a_lbl, a_pid),
        'MM':   (m_p1, m_p2, m_lbl, m_pid),
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

    passes = [
        ('frozen',    dict(is_training=False, shuffle_seed=None)),
        ('ttbn',      dict(is_training=True,  shuffle_seed=None)),
        ('ttbn_shuf', dict(is_training=True,  shuffle_seed=0)),
    ]

    results = {}   # results[pass][ds] = per-sample dict
    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt)
        print('Restored %s' % ckpt)
        for pass_name, kw in passes:
            results[pass_name] = {}
            for ds, (imgs, flows, _, _) in datasets.items():
                print('Scoring pass=%s dataset=%s (%d samples) ...' % (pass_name, ds, len(imgs)))
                results[pass_name][ds] = score_pass(
                    sess, tensors, imgs, flows,
                    batch_size=args.batch_size, **kw)

    # ── Patient-mean AUCs ─────────────────────────────────────────────────────
    summary = {'checkpoint': ckpt, 'batch_size': args.batch_size, 'auc': {}}
    for pass_name, _ in passes:
        summary['auc'][pass_name] = {}
        pooled = {k: {} for k in STREAMS}
        for ds in datasets:
            _, _, _, pids = datasets[ds]
            summary['auc'][pass_name][ds] = {}
            for k in STREAMS:
                pm = patient_mean(results[pass_name][ds][k], pids)
                pooled[k].update(pm)
                summary['auc'][pass_name][ds][k] = patient_auc(pm, label_by_pid)
        summary['auc'][pass_name]['POOLED'] = {
            k: patient_auc(pooled[k], label_by_pid) for k in STREAMS}

    # ── Report ────────────────────────────────────────────────────────────────
    hdr = ('pass', 'dataset') + STREAMS
    print('\n=== Patient-mean AUC (NOR vs disease) ===')
    print('%-10s %-8s' % hdr[:2] + ''.join('%12s' % k for k in STREAMS))
    for pass_name, _ in passes:
        for ds in ('ACDC', 'MM', 'POOLED'):
            row = summary['auc'][pass_name][ds]
            print('%-10s %-8s' % (pass_name, ds)
                  + ''.join('%12.4f' % row[k] for k in STREAMS))
    print('\n=== Delta vs frozen (TTBN - frozen) ===')
    for pass_name in ('ttbn', 'ttbn_shuf'):
        for ds in ('ACDC', 'MM', 'POOLED'):
            base = summary['auc']['frozen'][ds]
            row = summary['auc'][pass_name][ds]
            print('%-10s %-8s' % (pass_name, ds)
                  + ''.join('%+12.4f' % (row[k] - base[k]) for k in STREAMS))

    with open(os.path.join(args.out_dir, 'bn_tta_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        os.path.join(args.out_dir, 'bn_tta_scores.npz'),
        **{'%s__%s__%s' % (pn, ds, k): results[pn][ds][k]
           for pn in results for ds in results[pn] for k in STREAMS},
        acdc_pids=np.array(a_pid), acdc_labels=np.array(a_lbl),
        mm_pids=np.array(m_pid), mm_labels=np.array(m_lbl))
    print('\nSaved %s/bn_tta_summary.json and bn_tta_scores.npz' % args.out_dir)


if __name__ == '__main__':
    main()
