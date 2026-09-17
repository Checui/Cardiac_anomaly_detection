#!/usr/bin/env python
"""Varying the training with the score fixed: the GAN's training-side variants
re-scored on the held-out protocol (thesis ch. 4, sec:gan_training).

Every checkpoint below already exists; nothing is retrained. Each is scored with the
headline score fixed -- flow-SSIM (or its frame-prediction analogue), full frame,
middle-60%% slices, patient mean -- on ACDC-50 and M&Ms-Testing, plus M&Ms-Validation
for the honest-selection column, with a paired patient-cluster bootstrap against the
headline checkpoint. Three ablations, one table each in the thesis:

  the frame pair    es_ed (ED input, flow ED->ES; the headline) vs ed_es (ES input,
                    flow ES->ED) vs consecutive systolic pairs (next_frame_systole).
                    The consecutive-pair checkpoint is scored twice: on its own
                    systolic pairs (its training distribution, ~9x the samples per
                    patient) and on the ED->ES pair with ED as input (forward in time
                    like its training pairs, sharing every sample with the headline
                    so the paired interval applies).
  the pretext       optical-flow prediction (GAN_tf) vs future-frame prediction
                    (GAN_tf_rgb; the aux head predicts the target frame). For the
                    frame pretext the motion stream is 1 - SSIM(pred frame, target
                    frame) -- the metric-matched analogue of flow-SSIM -- next to the
                    MSE+gradient loss the network-free control of tab:gan_nomodel
                    used, and the network-free inter-frame statistics themselves.
  the flow teacher  Farneback vs the frozen biomechanics registration network
                    (--aux_source registration). CONFOUNDED: the registration runs
                    were trained on ACDC+MM only (no RECON), with N4 ON, for 100
                    epochs; they are scored at the pre-committed final epoch (100)
                    with their own preprocessing (N4 on) and their own teacher's
                    field as GT. Validation selection would be contaminated (ACDC-50
                    was their training-time validation set).

Sanity anchor: flow_es_ed/flow_ssim must reproduce ACDC 0.8125 / MM 0.7310 /
MM_VAL 0.699 (gan_revert_out/summary.json). Cross-check: rgb_ed_es_v1 with
MSE+gradient under top-20%% aggregation must reproduce tab:gan_nomodel's ACDC 0.850
(model) / 0.625 (no model).

Outputs gan_training_out/{summary.json, scores.npz}; per-sample scores are kept with
pids and slice indices for every (config, dataset) so tables can be re-derived
without another GPU pass.
"""
import sys
import os
import json
import argparse
import types
import time

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

# Cap every native thread pool to the CPUs actually granted (login node / PBS
# cgroup): TF, OpenCV (Farneback), SimpleITK (N4) and torch (registration teacher)
# each size their pool from the machine's core count and oversubscribe each other.
N_THREADS = int(os.environ.get('OMP_NUM_THREADS', '8'))
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
    os.environ.setdefault(_v, str(N_THREADS))
try:
    import cv2
    cv2.setNumThreads(N_THREADS)
except Exception:
    pass

import GAN_tf
import GAN_tf_rgb
import data_loader as dl_flow
import data_loader_rgb as dl_rgb
from utils import compute_flow_ssim_scores
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score

EPS = 1e-10
FLOW_STREAMS = ('appe', 'flow_l1', 'flow_ssim', 'mag_ssim')
RGB_STREAMS = ('appe', 'aux_msegrad', 'aux_mse', 'frame_ssim', 'raw_msegrad', 'raw_l1')

# (name, model_type, ckpt_subdir, epoch, frame_mode, eval_pairs, direction, teacher, n4)
# Order matters: it is the pass schedule. Flow configs first (one graph), with the
# slow, global-state-changing loads (N4 + registration) last; then the rgb graph.
CONFIGS = [
    ('flow_es_ed',       'flow', 'ACDC_MM_RECON_ES_ED_FLOW_NOR_v2_orient_spacing_added',
     50, 'es_ed', 'edes', 'ed', 'farneback', False),
    ('flow_sys_on_edes', 'flow', 'ACDC_MM_RECON_NEXT_FRAME_SYSTOLE_FLOW_NOR_v2_added',
     50, 'next_frame_systole', 'edes', 'ed', 'farneback', False),
    ('flow_ed_es',       'flow', 'ACDC_MM_RECON_ED_ES_FLOW_NOR_v2_added',
     50, 'ed_es', 'edes', 'es', 'farneback', False),
    ('flow_sys_on_sys',  'flow', 'ACDC_MM_RECON_NEXT_FRAME_SYSTOLE_FLOW_NOR_v2_added',
     50, 'next_frame_systole', 'systole', 'ed', 'farneback', False),
    ('reg_es_ed',        'flow', 'ACDC_MM_ES_ED_FLOW_NOR_REG_registration_es_ed',
     100, 'es_ed', 'edes', 'ed', 'registration', True),
    ('reg_ed_es',        'flow', 'ACDC_MM_ED_ES_FLOW_NOR_REG_registration_ed_es',
     100, 'ed_es', 'edes', 'es', 'registration', True),
    ('rgb_es_ed',        'rgb',  'ACDC_MM_RECON_ES_ED_RGB_NOR_v2_orient_spacing_added',
     50, 'es_ed', 'edes', 'ed', None, False),
    ('rgb_sys_on_edes',  'rgb',  'ACDC_MM_RECON_NEXT_FRAME_SYSTOLE_RGB_NOR_v2_orient_spacing_added',
     50, 'next_frame_systole', 'edes', 'ed', None, False),
    ('rgb_ed_es_v2',     'rgb',  'ACDC_MM_RECON_ED_ES_RGB_NOR_v2_orient_spacing_added',
     50, 'ed_es', 'edes', 'es', None, False),
    ('rgb_ed_es_v1',     'rgb',  'ACDC_MM_RECON_ED_ES_RGB_NOR_v1_orient_spacing_added',
     50, 'ed_es', 'edes', 'es', None, False),
    ('rgb_sys_on_sys',   'rgb',  'ACDC_MM_RECON_NEXT_FRAME_SYSTOLE_RGB_NOR_v2_orient_spacing_added',
     50, 'next_frame_systole', 'systole', 'ed', None, False),
]
CONFIG_FIELDS = ('name', 'model_type', 'ckpt_subdir', 'epoch', 'frame_mode',
                 'eval_pairs', 'direction', 'teacher', 'n4')
REF_ROW = 'flow_es_ed/flow_ssim'
# Extra aggregation rows (cross-check against tab:gan_nomodel, which used a
# top-20% slice mean on the rgb v1 checkpoint)
TOPK_ROWS = [('rgb_ed_es_v1', 'aux_msegrad'), ('rgb_ed_es_v1', 'raw_msegrad')]


def cfg_dict(c):
    return dict(zip(CONFIG_FIELDS, c))


def load_key(c):
    d = cfg_dict(c)
    return (d['model_type'], d['eval_pairs'], d['direction'], d['teacher'], d['n4'])


# ── Loader configuration (every global set on every call, so nothing leaks) ──────
def configure_loader(dl, direction, teacher, n4, args):
    dl.set_orientation_normalization(True, args.orient_params)
    dl.set_spacing_normalization(True, 1.5, 128, (2.0, 2.0))
    dl.set_n4_bias_correction(bool(n4), args.n4_shrink, args.n4_iterations, args.n4_levels)
    dl.set_edes_direction(direction)
    if dl is dl_flow:
        if teacher == 'registration':
            import registration_flow
            registration_flow.configure(reg_repo=args.reg_repo)
            dl.set_flow_backend('registration')
        else:
            dl.set_flow_backend('farneback')
    print('[loader] module=%s direction=%s teacher=%s n4=%s'
          % (dl.__name__, direction, teacher, 'ON' if n4 else 'off'))


def load_sets(dl, pairs, args, smoke=False):
    """One load of every evaluation set under the CURRENT loader state.

    Both loader modules return (inputs, targets, labels, pids, slice_idxs) with the
    same pid formats; ED/ES loaders give one sample per slice, the systolic
    consecutive-pair loaders one per (slice, t).
    Returns {ds: (inputs, targets, labels, pids, slice_idxs)}.
    """
    if pairs == 'edes':
        f_acdc = lambda: dl.load_acdc_test_val_ed_es_data(args.acdc_dir)
        f_mm = lambda d: dl.load_mm_validation_ed_es_data(d, args.mm_csv)
    elif pairs == 'systole':
        f_acdc = lambda: dl.load_acdc_test_val_data(args.acdc_dir, restrict_to_systole=True)
        f_mm = lambda d: dl.load_mm_validation_data(d, args.mm_csv, restrict_to_systole=True)
    else:
        raise ValueError(pairs)
    datasets = {}
    t0 = time.time()
    if not smoke:
        print('=== Loading ACDC test set (full 50 patients), pairs=%s ===' % pairs)
        (a_p1, a_p2, a_lbl, a_pid, a_slc,
         t_p1, t_p2, t_lbl, t_pid, t_slc) = f_acdc()
        if len(t_p1) > 0:
            a_p1 = np.concatenate([a_p1, t_p1], axis=0)
            a_p2 = np.concatenate([a_p2, t_p2], axis=0)
            a_lbl = list(a_lbl) + list(t_lbl)
            a_pid = list(a_pid) + list(t_pid)
            a_slc = list(a_slc) + list(t_slc)
        a_pid = ['ACDC_%s' % p for p in a_pid]
        datasets['ACDC'] = (a_p1, a_p2, list(a_lbl), a_pid, list(a_slc))

        print('=== Loading M&Ms Testing set, pairs=%s ===' % pairs)
        m_p1, m_p2, m_lbl, m_pid, m_slc = f_mm(args.mm_test_dir)
        m_pid = ['MM_%s' % p for p in m_pid]
        datasets['MM'] = (m_p1, m_p2, list(m_lbl), m_pid, list(m_slc))

    print('=== Loading M&Ms Validation set (selection only), pairs=%s ===' % pairs)
    v_p1, v_p2, v_lbl, v_pid, v_slc = f_mm(args.mm_val_dir)
    v_pid = ['MMVAL_%s' % p for p in v_pid]
    datasets['MM_VAL'] = (v_p1, v_p2, list(v_lbl), v_pid, list(v_slc))

    if smoke:
        n = args.smoke_n
        ds = datasets['MM_VAL']
        datasets['MM_VAL'] = tuple(x[:n] for x in ds)
    for ds, (p1, _, _, pids, _) in datasets.items():
        print('%s: %d samples / %d patients' % (ds, len(p1), len(set(pids))))
    print('[load] %.1f min' % ((time.time() - t0) / 60.0))
    return datasets


# ── Graphs ───────────────────────────────────────────────────────────────────────
def _ps_msegrad(pred, ref):
    """Per-sample MSE + gradient loss, the GAN's appearance / ED-prediction loss."""
    dy1, dx1 = tf.image.image_gradients(pred)
    dy0, dx0 = tf.image.image_gradients(ref)
    inten = tf.reduce_mean((pred - ref) ** 2, axis=[1, 2, 3])
    gradi = tf.reduce_mean(
        tf.abs(tf.abs(dy1) - tf.abs(dy0)) + tf.abs(tf.abs(dx1) - tf.abs(dx0)),
        axis=[1, 2, 3])
    return inten + gradi


def build_flow_graph():
    """Generator + per-sample losses of GAN_tf (as gan_revert_eval.py, no patch maps)."""
    tf.reset_default_graph()
    h = w = 128
    plh_frame_true = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    plh_flow_true = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    plh_is_training = tf.placeholder(tf.bool)
    scaled_frame_true = (plh_frame_true / 0.5) - 1.0
    plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
    output_opt, output_appe = GAN_tf.Generator(scaled_frame_true, plh_is_training, plh_dropout_prob)
    ps_loss_appe = _ps_msegrad(output_appe, scaled_frame_true)
    ps_loss_opt = tf.reduce_mean(tf.abs(output_opt - plh_flow_true), axis=[1, 2, 3])
    return {
        'frame': plh_frame_true, 'target': plh_flow_true, 'is_training': plh_is_training,
        'ps_appe': ps_loss_appe, 'ps_aux': ps_loss_opt, 'out_aux': output_opt,
    }


def build_rgb_graph():
    """Generator + per-sample losses of GAN_tf_rgb (ps_loss_ed of its training loop),
    plus the network-free inter-frame statistics computed in-graph on the same
    [-1, 1] scaling as notebook cell 32 (tab:gan_nomodel)."""
    tf.reset_default_graph()
    h = w = 128
    plh_frame_true = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    plh_target_true = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    plh_is_training = tf.placeholder(tf.bool)
    scaled_frame_true = (plh_frame_true / 0.5) - 1.0
    scaled_target_true = (plh_target_true / 0.5) - 1.0
    plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
    output_ed, output_appe = GAN_tf_rgb.Generator(scaled_frame_true, plh_is_training, plh_dropout_prob)
    ps_loss_appe = _ps_msegrad(output_appe, scaled_frame_true)
    ps_aux_msegrad = _ps_msegrad(output_ed, scaled_target_true)
    ps_aux_mse = tf.reduce_mean((output_ed - scaled_target_true) ** 2, axis=[1, 2, 3])
    ps_raw_msegrad = _ps_msegrad(scaled_frame_true, scaled_target_true)
    ps_raw_l1 = tf.reduce_mean(tf.abs(scaled_target_true - scaled_frame_true), axis=[1, 2, 3])
    return {
        'frame': plh_frame_true, 'target': plh_target_true, 'is_training': plh_is_training,
        'ps_appe': ps_loss_appe, 'ps_aux': ps_aux_msegrad, 'ps_aux_mse': ps_aux_mse,
        'ps_raw_msegrad': ps_raw_msegrad, 'ps_raw_l1': ps_raw_l1, 'out_aux': output_ed,
    }


def frame_ssim_scores(targets01, preds_pm1):
    """1 - SSIM(target frame, predicted frame), both in [0, 1]; data_range = max - min
    over both, channel_axis=-1 -- the convention of utils.compute_flow_ssim_scores."""
    n = len(targets01)
    out = np.zeros(n)
    for k in range(n):
        gt = np.asarray(targets01[k], dtype=np.float64)
        pr = np.clip(0.5 * (np.asarray(preds_pm1[k], dtype=np.float64) + 1.0), 0.0, 1.0)
        dr = float(np.max([gt, pr]) - np.min([gt, pr])) or 1.0
        out[k] = 1.0 - ssim(gt, pr, data_range=dr, channel_axis=-1)
    return out


def score_pass(sess, t, model_type, inputs, targets, batch_size):
    """Frozen-BN scoring pass; returns {stream: per-sample array}."""
    n = len(inputs)
    streams = FLOW_STREAMS if model_type == 'flow' else RGB_STREAMS
    out = {k: np.zeros(n) for k in streams}
    batches = np.array_split(np.arange(n), max(1, int(np.ceil(n / batch_size))))
    for vb in batches:
        feed = {t['frame']: inputs[vb], t['target']: targets[vb], t['is_training']: False}
        if model_type == 'flow':
            appe, aux, pred = sess.run([t['ps_appe'], t['ps_aux'], t['out_aux']], feed_dict=feed)
            out['appe'][vb] = appe
            out['flow_l1'][vb] = aux
            fs, ms = compute_flow_ssim_scores(targets[vb], pred)
            out['flow_ssim'][vb] = fs
            out['mag_ssim'][vb] = ms
        else:
            appe, aux, aux_mse, raw_mg, raw_l1, pred = sess.run(
                [t['ps_appe'], t['ps_aux'], t['ps_aux_mse'], t['ps_raw_msegrad'],
                 t['ps_raw_l1'], t['out_aux']], feed_dict=feed)
            out['appe'][vb] = appe
            out['aux_msegrad'][vb] = aux
            out['aux_mse'][vb] = aux_mse
            out['raw_msegrad'][vb] = raw_mg
            out['raw_l1'][vb] = raw_l1
            out['frame_ssim'][vb] = frame_ssim_scores(targets[vb], pred)
    return out


# ── Aggregation / AUC / bootstrap ────────────────────────────────────────────────
def group_by_patient(scores, pids):
    by = {}
    for s, p in zip(scores, pids):
        by.setdefault(p, []).append(float(s))
    return by


def patient_mean(scores, pids):
    return {p: float(np.mean(v)) for p, v in group_by_patient(scores, pids).items()}


def patient_topk_mean(scores, pids, top_k=0.2):
    """Notebook cell 28 topk_mean: mean of the top ceil(n*top_k) slice scores."""
    out = {}
    for p, v in group_by_patient(scores, pids).items():
        arr = np.sort(np.asarray(v))
        k = max(1, int(np.ceil(len(arr) * top_k)))
        out[p] = float(arr[-k:].mean())
    return out


def patient_auc(score_by_pid, label_by_pid, pids=None):
    pids = sorted(score_by_pid) if pids is None else list(pids)
    y = [0 if label_by_pid[p] == 'NOR' else 1 for p in pids]
    s = [score_by_pid[p] for p in pids]
    if len(set(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, s))


def paired_bootstrap(ref_by, row_by, label_by_pid, n_boot=2000, seed=0):
    """Patient-cluster bootstrap of the row AUC and of (row - ref) on the patients
    common to both. The same seed for every row means rows that share a patient set
    also share the resamples, so their deltas are mutually paired too."""
    ps = sorted(set(ref_by) & set(row_by))
    y = np.array([0 if label_by_pid[p] == 'NOR' else 1 for p in ps])
    r = np.array([ref_by[p] for p in ps])
    s = np.array([row_by[p] for p in ps])
    rng = np.random.RandomState(seed)
    n = len(ps)
    aucs, deltas = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        a = roc_auc_score(yy, s[idx])
        aucs.append(a)
        deltas.append(a - roc_auc_score(yy, r[idx]))
    if not aucs:
        nan = float('nan')
        return {'n_common': n, 'ci': [nan, nan],
                'delta_vs_headline': {'mean': nan, 'ci': [nan, nan], 'frac_gt0': nan}}
    a = np.array(aucs)
    d = np.array(deltas)
    return {
        'n_common': n,
        'ci': [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
        'delta_vs_headline': {
            'mean': float(d.mean()),
            'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            'frac_gt0': float((d > 0).mean()),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_root', default='./training_saver')
    ap.add_argument('--acdc_dir', default='../Dataset_2')
    ap.add_argument('--mm_test_dir', default='../Dataset_1/Testing')
    ap.add_argument('--mm_val_dir', default='../Dataset_1/Validation')
    ap.add_argument('--mm_csv', default='../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
    ap.add_argument('--orient_params', default='../reconstructed_sax_images_training_2023/segmentation/orientation_params.csv')
    ap.add_argument('--reg_repo', default='../biomechanics-cardiac-motion-hpc',
                    help='registration-teacher repo (registration_flow.py default path does not exist)')
    ap.add_argument('--n4_shrink', type=int, default=4)
    ap.add_argument('--n4_iterations', type=int, default=50)
    ap.add_argument('--n4_levels', type=int, default=4)
    ap.add_argument('--top_k', type=float, default=0.2)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--out_dir', default='./gan_training_out')
    ap.add_argument('--only', nargs='*', default=None,
                    help='subset of config names to run (debugging)')
    ap.add_argument('--smoke', action='store_true',
                    help='M&Ms Validation only, truncated, 50 bootstrap reps, every '
                         'config and both graphs; end-to-end sanity check before qsub')
    ap.add_argument('--smoke_n', type=int, default=64)
    args = ap.parse_args()
    if args.smoke:
        args.n_boot = 50
    os.makedirs(args.out_dir, exist_ok=True)

    configs = [c for c in CONFIGS if args.only is None or c[0] in args.only]
    config_gpu = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=N_THREADS,
                                inter_op_parallelism_threads=2)
    config_gpu.gpu_options.allow_growth = True

    results = {}      # results[config][ds][stream] -> per-sample array
    meta = {}         # meta[config][ds] -> (labels, pids, slice_idxs)
    checkpoints = {}
    mus = {}
    label_by_pid = {}
    t_start = time.time()

    current_key, datasets = None, None
    current_model, sess, tensors, saver = None, None, None, None
    ref_meta = None   # (ds -> (pids, slcs)) of the first ED/ES flow load, for sanity checks

    try:
        for c in configs:
            d = cfg_dict(c)
            name = d['name']
            dl = dl_flow if d['model_type'] == 'flow' else dl_rgb

            # ── Graph / session per model type ─────────────────────────────────
            if d['model_type'] != current_model:
                if sess is not None:
                    sess.close()
                tensors = build_flow_graph() if d['model_type'] == 'flow' else build_rgb_graph()
                saver = tf.train.Saver(var_list=tf.global_variables())
                sess = tf.Session(config=config_gpu)
                current_model = d['model_type']
                print('\n##### %s graph built #####' % current_model)

            # ── Data (re)load when the loader key changes ──────────────────────
            key = load_key(c)
            if key != current_key:
                del datasets
                datasets = None
                configure_loader(dl, d['direction'], d['teacher'], d['n4'], args)
                datasets = load_sets(dl, d['eval_pairs'], args, smoke=args.smoke)
                current_key = key
                if d['model_type'] == 'flow' and d['eval_pairs'] == 'edes':
                    cur = {ds: (list(v[3]), list(v[4])) for ds, v in datasets.items()}
                    if ref_meta is None:
                        ref_meta = cur
                    else:
                        for ds in cur:
                            same = cur[ds] == ref_meta[ds]
                            print('[check] %s sample set identical to headline load: %s'
                                  % (ds, same))
                            if not same:
                                print('WARNING %s: %d vs %d samples, %d vs %d patients'
                                      % (ds, len(cur[ds][0]), len(ref_meta[ds][0]),
                                         len(set(cur[ds][0])), len(set(ref_meta[ds][0]))))

            # ── Restore checkpoint and score every set ─────────────────────────
            ckpt_dir = os.path.join(args.ckpt_root, d['ckpt_subdir'])
            ckpt = os.path.join(ckpt_dir, 'model_ckpt_%d.ckpt' % d['epoch'])
            saver.restore(sess, ckpt)
            checkpoints[name] = ckpt
            mu_path = os.path.join(ckpt_dir, 'mu_baseline_%d.json' % d['epoch'])
            if os.path.exists(mu_path):
                with open(mu_path) as f:
                    mus[name] = json.load(f)
            print('\n[%s] restored %s' % (name, ckpt))
            results[name], meta[name] = {}, {}
            for ds, (p1, p2, lbls, pids, slcs) in datasets.items():
                for p, l in zip(pids, lbls):
                    prev = label_by_pid.setdefault(p, l)
                    assert prev == l, 'inconsistent label for %s' % p
                t0 = time.time()
                results[name][ds] = score_pass(sess, tensors, d['model_type'], p1, p2,
                                               batch_size=args.batch_size)
                meta[name][ds] = (list(lbls), list(pids), list(slcs))
                bad = {k: int(np.isnan(v).sum()) for k, v in results[name][ds].items() if np.isnan(v).any()}
                print('  scored %s: %d samples in %.1f min%s'
                      % (ds, len(p1), (time.time() - t0) / 60.0,
                         ('  NaN: %s' % bad) if bad else ''))
            print('[elapsed] %.1f min' % ((time.time() - t_start) / 60.0))
    finally:
        if sess is not None:
            sess.close()

    # ── Rows: (row, config, stream, agg) ──────────────────────────────────────
    rows = []
    for name in results:
        for stream in results[name][next(iter(results[name]))]:
            rows.append(('%s/%s' % (name, stream), name, stream, 'patient_mean'))
    for name, stream in TOPK_ROWS:
        if name in results:
            rows.append(('%s/%s/top20' % (name, stream), name, stream, 'patient_top20'))
    assert REF_ROW in [r[0] for r in rows] or args.only is not None, 'reference row missing'

    ds_names = [ds for ds in ('ACDC', 'MM', 'MM_VAL')
                if all(ds in results[n] for n in results)]

    def row_repr(row):
        _, name, stream, agg = row
        out = {}
        for ds in ds_names:
            s = results[name][ds][stream]
            pids = meta[name][ds][1]
            out[ds] = (patient_mean(s, pids) if agg == 'patient_mean'
                       else patient_topk_mean(s, pids, args.top_k))
        return out

    reprs = {row[0]: row_repr(row) for row in rows}
    summary = {
        'configs': [cfg_dict(c) for c in configs],
        'checkpoints': checkpoints, 'mu': mus,
        'rows': [list(r) for r in rows], 'ref_row': REF_ROW,
        'batch_size': args.batch_size, 'n_boot': args.n_boot, 'top_k': args.top_k,
        'smoke': args.smoke,
        'n_samples': {n: {ds: len(meta[n][ds][1]) for ds in meta[n]} for n in meta},
        'n_patients': {n: {ds: len(set(meta[n][ds][1])) for ds in meta[n]} for n in meta},
        'auc': {ds: {} for ds in ds_names},
        'bootstrap': {ds: {} for ds in ds_names},
    }
    for ds in ds_names:
        for row_name, by in reprs.items():
            summary['auc'][ds][row_name] = patient_auc(by[ds], label_by_pid)
        if ds == 'MM_VAL' and not args.smoke:
            continue
        if REF_ROW not in reprs:
            continue
        print('Bootstrapping %s (%d reps, %d rows) ...' % (ds, args.n_boot, len(reprs)))
        ref_by = reprs[REF_ROW][ds]
        for row_name, by in reprs.items():
            summary['bootstrap'][ds][row_name] = paired_bootstrap(
                ref_by, by[ds], label_by_pid, n_boot=args.n_boot)

    # ── Report ────────────────────────────────────────────────────────────────
    print('\n=== Varying the training, score fixed: patient-level AUC (NOR vs disease) ===')
    print('%-34s' % 'row' + ''.join('%22s' % ds for ds in ds_names))
    for row_name, _, _, _ in rows:
        line = '%-34s' % row_name
        for ds in ds_names:
            auc = summary['auc'][ds][row_name]
            b = summary['bootstrap'][ds].get(row_name)
            if b:
                line += '%9.4f [%.2f,%.2f]' % (auc, b['ci'][0], b['ci'][1])
            else:
                line += '%22.4f' % auc
        print(line)
    print('\n=== Paired bootstrap delta vs %s (mean [95%% CI], frac>0, n_common) ===' % REF_ROW)
    for row_name, _, _, _ in rows:
        if row_name == REF_ROW:
            continue
        line = '%-34s' % row_name
        for ds in ds_names:
            b = summary['bootstrap'][ds].get(row_name)
            if not b:
                continue
            dd = b['delta_vs_headline']
            line += '   %+0.4f [%+0.3f,%+0.3f] %4.0f%% n=%d' % (
                dd['mean'], dd['ci'][0], dd['ci'][1], 100 * dd['frac_gt0'], b['n_common'])
        print(line)

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    arrays = {}
    for n in results:
        for ds in results[n]:
            for k, v in results[n][ds].items():
                arrays['%s__%s__%s' % (n, ds, k)] = v
            arrays['%s__%s__pids' % (n, ds)] = np.array(meta[n][ds][1])
            arrays['%s__%s__labels' % (n, ds)] = np.array(meta[n][ds][0])
            arrays['%s__%s__slice_idx' % (n, ds)] = np.array(meta[n][ds][2])
    np.savez_compressed(os.path.join(args.out_dir, 'scores.npz'), **arrays)
    print('\nSaved %s/summary.json and scores.npz  (total %.1f min)'
          % (args.out_dir, (time.time() - t_start) / 60.0))


if __name__ == '__main__':
    main()
