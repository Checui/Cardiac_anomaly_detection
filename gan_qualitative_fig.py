#!/usr/bin/env python
"""Extract the panels for the thesis qualitative figure (fig:gan_qualitative).

Runs the frozen baseline flow-GAN checkpoint
(ACDC_MM_RECON_ES_ED_FLOW_NOR_v2_orient_spacing_added, epoch 50 -- same as
gan_flowmetric_eval.py) on the ACDC test set only, and saves, for one healthy
and one DCM patient (the median patient of each group by patient-mean flow-SSIM,
so neither is cherry-picked): the input ED frame, the reconstruction, and the
ground-truth and predicted ED->ES flow fields for every loaded slice.

Rendering is a separate step (render_gan_qualitative.py) so the figure can be
iterated on without re-running TF. Output: gan_flowmetric_out/qualitative_panels.npz
"""
import sys
import os
import types

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

CKPT_DIR = './training_saver/ACDC_MM_RECON_ES_ED_FLOW_NOR_v2_orient_spacing_added'
EPOCH = 50
ACDC_DIR = '../Dataset_2'
ORIENT_PARAMS = '../reconstructed_sax_images_training_2023/segmentation/orientation_params.csv'
OUT = './gan_flowmetric_out/qualitative_panels.npz'
# Median patient of each group by patient-mean flow-SSIM (from gan_flowmetric_out/scores.npz)
PATIENTS = {'NOR': 'patient139', 'DCM': 'patient117'}


def main():
    # Loader config: the baseline checkpoint's recipe (as in gan_flowmetric_eval.py)
    dl_flow.set_edes_direction('ed')
    dl_flow.set_orientation_normalization(True, ORIENT_PARAMS)
    dl_flow.set_spacing_normalization(True, 1.5, 128, (2.0, 2.0))
    dl_flow.set_n4_bias_correction(False)

    print('=== Loading ACDC test set (full 50 patients) ===')
    (a_p1, a_p2, a_lbl, a_pid, a_slc,
     t_p1, t_p2, t_lbl, t_pid, t_slc) = dl_flow.load_acdc_test_val_ed_es_data(ACDC_DIR)
    if len(t_p1) > 0:
        a_p1 = np.concatenate([a_p1, t_p1], axis=0)
        a_p2 = np.concatenate([a_p2, t_p2], axis=0)
        a_lbl = list(a_lbl) + list(t_lbl)
        a_pid = list(a_pid) + list(t_pid)
        a_slc = list(a_slc) + list(t_slc)
    a_pid = np.array(a_pid)
    a_slc = np.array(a_slc)
    print('ACDC: %d samples / %d patients' % (len(a_p1), len(set(a_pid.tolist()))))

    keep = np.isin(a_pid, list(PATIENTS.values()))
    imgs, flows = a_p1[keep], a_p2[keep]
    pids, slcs = a_pid[keep], a_slc[keep]
    lbls = np.array(a_lbl)[keep]
    print('Selected %d slices: %s' % (len(imgs),
          {p: int((pids == p).sum()) for p in PATIENTS.values()}))

    tf.reset_default_graph()
    h, w = imgs.shape[1:3]
    plh_frame_true = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    plh_is_training = tf.placeholder(tf.bool)
    scaled_frame_true = (plh_frame_true / 0.5) - 1.0
    plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
    output_opt, output_appe = GAN_tf.Generator(scaled_frame_true, plh_is_training,
                                               plh_dropout_prob)

    ckpt = os.path.join(CKPT_DIR, 'model_ckpt_%d.ckpt' % EPOCH)
    saver = tf.train.Saver(var_list=tf.global_variables())
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt)
        print('Restored %s' % ckpt)
        pred_flow, recon = sess.run(
            [output_opt, output_appe],
            feed_dict={plh_frame_true: imgs, plh_is_training: False})

    fs, ms = compute_flow_ssim_scores(flows, pred_flow)
    for p in PATIENTS.values():
        m = pids == p
        print('%s: per-slice 1-SSIM(flow) %s  mean=%.4f' %
              (p, np.round(fs[m], 3).tolist(), fs[m].mean()))
        print('  mean |GT| mag %.3f px, mean |pred| mag %.3f px' %
              (flows[m][..., 2].mean(), pred_flow[m][..., 2].mean()))

    np.savez_compressed(
        OUT,
        images=imgs.astype(np.float32), recon=recon.astype(np.float32),
        gt_flow=flows.astype(np.float32), pred_flow=pred_flow.astype(np.float32),
        pids=pids, slices=slcs, labels=lbls, flow_ssim=fs, mag_ssim=ms)
    print('Saved %s' % OUT)


if __name__ == '__main__':
    main()
