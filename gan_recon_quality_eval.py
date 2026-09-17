#!/usr/bin/env python
"""Reconstruction-quality pass on the baseline GAN: how well does the appearance head
reconstruct its input frame, per slice / patient / scanner vendor?

Scores the frozen baseline flow-GAN checkpoint
(ACDC_MM_RECON_ES_ED_FLOW_NOR_v2_orient_spacing_added, epoch 50) on held-out ACDC-50 and
M&Ms-Testing (+ M&Ms-Validation), exactly as gan_flowmetric_eval.py does, and additionally
fetches the reconstruction head's output (`output_appe`) so that per-slice reconstruction
QUALITY metrics can be computed between the reconstruction and the input frame:

  recon_mse         mean squared error on the loader's [0,1] frame scale
  recon_psnr        10*log10(1 / MSE), MAX = 1  (the repo's own convention, utils.py:209)
  recon_ssim_dr1    skimage SSIM, data_range = 1.0 (7x7 uniform window)
  recon_ssim_joint  skimage SSIM, data_range = joint max-min (mirrors utils.compute_flow_ssim_scores)
  recon_ssim_crop   SSIM (data_range 1.0) on the central crop x crop px box -- frames are
                    LV-centred by orientation normalisation, so at 1.5 mm/px a 64 px crop
                    is a 96 mm heart box; full-frame SSIM is inflated by the zero background

plus the sanity-anchor anomaly streams (flow_ssim, mag_ssim, flow_l1, appe) which must
reproduce the known patient-mean AUCs (ACDC 0.8125 / MM 0.7310; appe 0.5325 / 0.4920).

The reconstruction is `tanh`-bounded in [-1, 1] (input scaled as x/0.5 - 1 inside the
graph), so it is mapped back with (r + 1) / 2 before any metric; all frames have three
identical channels, channel 0 is used.

Outputs (--out_dir): summary.json, scores.npz ({ds}__{stream} + {ds}__pids/labels/slice_idx,
middle-60% slices only) and, with --save_recon, recon.npz (float16 single-channel
reconstructions + inputs, same row order, pids + slice_idx repeated). Downstream:
gan_vendor_table.py --recon <out_dir>/scores.npz  (joins on (pid, slice_idx)).
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
from skimage.metrics import structural_similarity as ssim
import GAN_tf
import data_loader as dl_flow
from utils import compute_flow_ssim_scores
from qfae_report import _patient_auc

ANCHOR_STREAMS = ('flow_ssim', 'mag_ssim', 'flow_l1', 'appe')
QUALITY_STREAMS = ('recon_mse', 'recon_psnr', 'recon_ssim_dr1', 'recon_ssim_joint', 'recon_ssim_crop')
STREAMS = ANCHOR_STREAMS + QUALITY_STREAMS
# quality metric -> sign that turns it into an anomaly score (higher = more anomalous)
QUALITY_AS_SCORE_SIGN = {'recon_mse': +1.0, 'recon_psnr': -1.0, 'recon_ssim_dr1': -1.0,
                         'recon_ssim_joint': -1.0, 'recon_ssim_crop': -1.0}
ANCHORS_EXPECTED = {('ACDC', 'flow_ssim'): 0.8125, ('MM', 'flow_ssim'): 0.7310,
                    ('ACDC', 'appe'): 0.5325, ('MM', 'appe'): 0.4920}
ANCHOR_TOL = 5e-4


def recon_quality(img, recon, crop=64):
    """Per-slice quality of the reconstruction `recon` (tanh, [-1,1]) against `img` ([0,1]).

    Returns (dict of QUALITY_STREAMS, recon01 as float16 (H, W)).
    """
    x = img[..., 0].astype(np.float64)
    r = np.clip((recon[..., 0].astype(np.float64) + 1.0) / 2.0, 0.0, 1.0)
    mse = float(np.mean((x - r) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-10)))
    s_dr1 = float(ssim(x, r, data_range=1.0))
    dr = float(max(x.max(), r.max()) - min(x.min(), r.min())) or 1.0
    s_joint = float(ssim(x, r, data_range=dr))
    h, w = x.shape
    c = int(crop)
    r0, c0 = (h - c) // 2, (w - c) // 2
    s_crop = float(ssim(x[r0:r0 + c, c0:c0 + c], r[r0:r0 + c, c0:c0 + c], data_range=1.0))
    return ({'recon_mse': mse, 'recon_psnr': psnr, 'recon_ssim_dr1': s_dr1,
             'recon_ssim_joint': s_joint, 'recon_ssim_crop': s_crop},
            r.astype(np.float16))


def score_pass(sess, t, images, flows, batch_size, crop, save_recon):
    """Frozen-BN scoring pass; per-sample score dict (+ float16 recon / input frames)."""
    n = len(images)
    out = {k: np.zeros(n) for k in STREAMS}
    h, w = images.shape[1:3]
    recon16 = np.zeros((n, h, w), dtype=np.float16) if save_recon else None
    input16 = np.zeros((n, h, w), dtype=np.float16) if save_recon else None
    batches = np.array_split(np.arange(n), max(1, int(np.ceil(n / batch_size))))
    for vb in batches:
        appe, l1, pred_flow, recon = sess.run(
            [t['ps_appe'], t['ps_opt'], t['out_opt'], t['out_appe']],
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
            q, r16 = recon_quality(images[i], recon[j], crop=crop)
            for k in QUALITY_STREAMS:
                out[k][i] = q[k]
            if save_recon:
                recon16[i] = r16
                input16[i] = images[i][..., 0].astype(np.float16)
    return out, recon16, input16


def load_datasets(args, smoke=False):
    """One load of every evaluation set under the CURRENT loader state.

    Returns {ds: (images, flows, labels, pids, slice_idxs)}.  (Copied from
    gan_revert_eval.load_datasets so slice indices are kept.)
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

    print('=== Loading M&Ms Validation set ===')
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
    ap.add_argument('--crop', type=int, default=64, help='side (px) of the central crop for recon_ssim_crop')
    ap.add_argument('--save_recon', action='store_true',
                    help='also save float16 single-channel reconstructions + inputs (recon.npz)')
    ap.add_argument('--out_dir', default='./gan_recon_quality_out')
    ap.add_argument('--smoke', action='store_true',
                    help='M&Ms Validation only, truncated, 50 bootstrap reps; end-to-end check before qsub')
    ap.add_argument('--smoke_n', type=int, default=48)
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

    datasets = load_datasets(args, smoke=args.smoke)
    label_by_pid = {}
    for _, (_, _, lbls, pids, _) in datasets.items():
        for p, l in zip(pids, lbls):
            prev = label_by_pid.setdefault(p, l)
            assert prev == l, 'inconsistent label for %s' % p

    # ── Graph: Generator + per-sample losses only (no D, no optimizer) ────────
    tf.reset_default_graph()
    any_imgs = next(iter(datasets.values()))[0]
    h, w = any_imgs.shape[1:3]
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
        'ps_appe': ps_loss_appe, 'ps_opt': ps_loss_opt,
        'out_opt': output_opt, 'out_appe': output_appe,
    }

    ckpt = os.path.join(args.ckpt_dir, 'model_ckpt_%d.ckpt' % args.epoch)
    saver = tf.train.Saver(var_list=tf.global_variables())
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    results, recons, inputs = {}, {}, {}
    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt)
        print('Restored %s' % ckpt)
        for ds, (imgs, flows, _, _, _) in datasets.items():
            print('Scoring dataset=%s (%d samples) ...' % (ds, len(imgs)))
            results[ds], recons[ds], inputs[ds] = score_pass(
                sess, tensors, imgs, flows, batch_size=args.batch_size,
                crop=args.crop, save_recon=args.save_recon)

    # ── Patient-mean AUCs (anchor streams; quality streams sign-flipped as scores) ──
    summary = {'checkpoint': ckpt, 'batch_size': args.batch_size, 'n_boot': args.n_boot,
               'crop': args.crop, 'smoke': bool(args.smoke),
               'n_samples': {}, 'auc': {}, 'auc_quality_as_score': {},
               'quality_as_score_sign': QUALITY_AS_SCORE_SIGN, 'quality': {}}
    for ds, (_, _, lbls, pids, _) in datasets.items():
        summary['n_samples'][ds] = {'slices': len(pids), 'patients': len(set(pids))}
        pids_a, lbls_a = np.array(pids), np.array(lbls)
        summary['auc'][ds] = {}
        for k in ANCHOR_STREAMS:
            summary['auc'][ds][k] = _patient_auc(results[ds][k], pids_a, lbls_a, n_boot=args.n_boot)
        summary['auc_quality_as_score'][ds] = {}
        summary['quality'][ds] = {}
        for k in QUALITY_STREAMS:
            summary['auc_quality_as_score'][ds][k] = _patient_auc(
                QUALITY_AS_SCORE_SIGN[k] * results[ds][k], pids_a, lbls_a, n_boot=args.n_boot)
            # patient means, NOR vs disease
            by = {}
            for s, p in zip(results[ds][k], pids):
                by.setdefault(p, []).append(s)
            pm = {p: float(np.mean(v)) for p, v in by.items()}
            nor = [pm[p] for p in pm if label_by_pid[p] == 'NOR']
            dis = [pm[p] for p in pm if label_by_pid[p] != 'NOR']
            summary['quality'][ds][k] = {
                'nor_mean': float(np.mean(nor)) if nor else None, 'nor_sd': float(np.std(nor, ddof=1)) if len(nor) > 1 else None,
                'dis_mean': float(np.mean(dis)) if dis else None, 'dis_sd': float(np.std(dis, ddof=1)) if len(dis) > 1 else None,
                'n_nor': len(nor), 'n_dis': len(dis),
                'slice_min': float(np.min(results[ds][k])), 'slice_max': float(np.max(results[ds][k]))}

    # ── Report ────────────────────────────────────────────────────────────────
    print('\n=== Patient-mean AUC (NOR vs disease), anchor streams ===')
    for ds in datasets:
        print('  %-7s' % ds + ''.join('  %s %.4f [%.2f,%.2f]' % (
            k, summary['auc'][ds][k]['auc'], summary['auc'][ds][k]['lo'], summary['auc'][ds][k]['hi'])
            for k in ANCHOR_STREAMS))
    print('\n=== Reconstruction quality, patient mean: NOR vs disease ===')
    for ds in datasets:
        for k in QUALITY_STREAMS:
            q = summary['quality'][ds][k]
            print('  %-7s %-17s NOR %s (n=%d)   disease %s (n=%d)   as-score AUC %.4f' % (
                ds, k,
                'nan' if q['nor_mean'] is None else '%.4f' % q['nor_mean'], q['n_nor'],
                'nan' if q['dis_mean'] is None else '%.4f' % q['dis_mean'], q['n_dis'],
                summary['auc_quality_as_score'][ds][k]['auc']))

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    arrays = {}
    for ds, (_, _, lbls, pids, slcs) in datasets.items():
        for k in STREAMS:
            arrays['%s__%s' % (ds, k)] = results[ds][k]
        arrays['%s__pids' % ds] = np.array(pids)
        arrays['%s__labels' % ds] = np.array(lbls)
        arrays['%s__slice_idx' % ds] = np.array(slcs, dtype=np.int64)
    np.savez_compressed(os.path.join(args.out_dir, 'scores.npz'), **arrays)
    if args.save_recon:
        rec = {}
        for ds, (_, _, _, pids, slcs) in datasets.items():
            rec['recon__%s' % ds] = recons[ds]
            rec['input__%s' % ds] = inputs[ds]
            rec['%s__pids' % ds] = np.array(pids)
            rec['%s__slice_idx' % ds] = np.array(slcs, dtype=np.int64)
        np.savez_compressed(os.path.join(args.out_dir, 'recon.npz'), **rec)
        print('Saved recon.npz')
    print('Saved %s/summary.json and scores.npz' % args.out_dir)

    # ── Anchor gate (after saving; skipped under --smoke) ─────────────────────
    if args.smoke:
        print('[GATE] skipped (--smoke)')
        return
    ok_all = True
    for (ds, k), exp in ANCHORS_EXPECTED.items():
        got = summary['auc'][ds][k]['auc']
        ok = abs(got - exp) < ANCHOR_TOL
        ok_all &= ok
        print('  [%s] %s %s = %.4f (expected %.4f)' % ('OK ' if ok else 'BAD', ds, k, got, exp))
    print('[GATE] %s' % ('OK' if ok_all else '** MISMATCH ** -- wrong checkpoint / loader recipe?'))


if __name__ == '__main__':
    main()
