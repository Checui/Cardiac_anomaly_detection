"""Per-sample encoder feature-map (embedding) visualisation for two flow-GAN
checkpoints — the 20-epoch L1 baseline vs the L1+SSIM mix — on ACDC test samples.

For each selected sample (one NOR, one DCM patient by default; mid-slice), runs
the frame through both restored generators and renders one figure:
    rows    = [L1 baseline, L1+SSIM mix, difference (mix - base)]
    columns = [context, h0, h1, h2, h3, h4, h5-bottleneck]
Feature maps are channel-mean |activation|, each layer normalised by the joint
max over the two models so the two rows are directly comparable; the difference
row is signed in those shared units. Context column: input ED frame / GT flow
magnitude / predicted-flow-magnitude difference.

Caveat: the two nets were trained from independent inits, so channel bases are
not aligned — channel-mean |act| is basis-sensitive; read the difference row
qualitatively (is the spatial saliency the same?), not as a per-pixel metric.

Loader config mirrors the two training jobs (es_ed, orient+spacing, Farneback,
no N4). CPU-only; run via qsub (submit_embmap.pbs).
"""
import sys
import os
import types
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class MockProgressBar:
    FULL = 'full'

    def __init__(self, n, fmt='full'):
        self.n = n
        self.current = 0

    def __call__(self):
        pass

    def done(self):
        pass


mock_pb_module = types.ModuleType("ProgressBar")
mock_pb_module.ProgressBar = MockProgressBar
sys.modules["ProgressBar"] = mock_pb_module

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.modules['tensorflow'] = tf

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import GAN_tf
import data_loader as dl_flow

CKPTS = {
    'L1 baseline': 'training_saver/ACDC_MM_RECON_ES_ED_FLOW_NOR_l1_baseline_20ep_v1/model_ckpt_20.ckpt',
    'L1+SSIM mix': 'training_saver/ACDC_MM_RECON_ES_ED_FLOW_NOR_SSIM_ssim_l1_mix_20ep_v1/model_ckpt_20.ckpt',
}
LAYER_TITLES = [
    'h0  128²×64', 'h1  128²×64', 'h2  64²×128',
    'h3  32²×256', 'h4  16²×512', 'h5 bottleneck  8²×512',
]


def channel_mean_abs(act):
    """(H, W, C) activations -> (H, W) channel-mean |activation| map."""
    return np.mean(np.abs(act), axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acdc_dir', default='../Dataset_2')
    ap.add_argument('--orient_params',
                    default='../reconstructed_sax_images_training_2023/segmentation/orientation_params.csv')
    ap.add_argument('--labels', nargs='+', default=['NOR', 'DCM'],
                    help='One sample (first patient, mid slice) is drawn per label.')
    ap.add_argument('--out_dir', default='embedding_maps')
    args = ap.parse_args()

    # Loader config identical to the two 20-epoch training jobs
    dl_flow.set_edes_direction('ed')                      # es_ed: ED frame is the model input
    dl_flow.set_flow_backend('farneback')
    dl_flow.set_orientation_normalization(True, args.orient_params)
    dl_flow.set_spacing_normalization(True, 1.5, 128, (2.0, 2.0))

    print('Loading ACDC test set (val+test halves) ...')
    (v_p1, v_p2, v_lab, v_pid, v_slc,
     t_p1, t_p2, t_lab, t_pid, t_slc) = dl_flow.load_acdc_test_val_ed_es_data(args.acdc_dir)
    images = np.concatenate([v_p1, t_p1], axis=0)
    flows = np.concatenate([v_p2, t_p2], axis=0)
    labels = list(v_lab) + list(t_lab)
    pids = list(v_pid) + list(t_pid)
    slcs = list(v_slc) + list(t_slc)
    print(f'{len(images)} samples / {len(set(pids))} patients loaded')

    sel = []
    for want in args.labels:
        idxs = [i for i, l in enumerate(labels) if l == want]
        if not idxs:
            print(f'[WARN] no samples with label {want}; skipping')
            continue
        pid0 = pids[idxs[0]]
        pidx = sorted([i for i in idxs if pids[i] == pid0], key=lambda i: slcs[i])
        sel.append(pidx[len(pidx) // 2])
    if not sel:
        sys.exit('no samples selected')
    for i in sel:
        print(f'selected: {pids[i]}  label={labels[i]}  slice={slcs[i]}')
    batch = images[sel]
    batch_flows = flows[sel]

    feats = {}
    for name, ckpt in CKPTS.items():
        tf.reset_default_graph()
        plh = tf.placeholder(tf.float32, [None, 128, 128, 3])
        plh_train = tf.placeholder(tf.bool)
        scaled = (plh / 0.5) - 1.0
        out_flow, out_frame, layers = GAN_tf.Generator(scaled, plh_train, 1.0, return_layers=True)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, ckpt)
            res = sess.run(layers[:6] + [out_flow],
                           feed_dict={plh: batch, plh_train: False})
        feats[name] = {'layers': res[:6], 'pred_flow': res[6]}
        print(f'restored + ran: {name}  ({ckpt})')

    os.makedirs(args.out_dir, exist_ok=True)
    base_name, mix_name = list(CKPTS)
    ink = '#444444'

    for k, si in enumerate(sel):
        fig, axes = plt.subplots(3, 7, figsize=(19, 8.2))
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        # ── context column ──
        axes[0, 0].imshow(batch[k, :, :, 0], cmap='gray')
        axes[0, 0].set_title('input ED frame', fontsize=9, color=ink)
        axes[1, 0].imshow(batch_flows[k, :, :, 2], cmap='viridis')
        axes[1, 0].set_title('GT flow magnitude', fontsize=9, color=ink)
        pf_base = feats[base_name]['pred_flow'][k, :, :, 2]
        pf_mix = feats[mix_name]['pred_flow'][k, :, :, 2]
        dmag = pf_mix - pf_base
        lim = max(np.abs(dmag).max(), 1e-6)
        axes[2, 0].imshow(dmag, cmap='RdBu_r', vmin=-lim, vmax=lim)
        axes[2, 0].set_title(f'pred flow-mag Δ  ±{lim:.2f}px', fontsize=9, color=ink)

        # ── feature-map columns ──
        print(f'\n=== {pids[si]} ({labels[si]}, slice {slcs[si]}) '
              f'relative embedding difference per layer ===')
        for c in range(6):
            a = channel_mean_abs(feats[base_name]['layers'][c][k])
            b = channel_mean_abs(feats[mix_name]['layers'][c][k])
            vmax = max(a.max(), b.max(), 1e-6)
            im0 = axes[0, c + 1].imshow(a / vmax, cmap='viridis', vmin=0, vmax=1)
            axes[1, c + 1].imshow(b / vmax, cmap='viridis', vmin=0, vmax=1)
            d = (b - a) / vmax
            dlim = max(np.abs(d).max(), 1e-6)
            imd = axes[2, c + 1].imshow(d, cmap='RdBu_r', vmin=-dlim, vmax=dlim)
            axes[0, c + 1].set_title(LAYER_TITLES[c], fontsize=9, color=ink)
            axes[2, c + 1].set_title(f'Δ ±{dlim:.2f}', fontsize=8, color=ink)
            rel = np.mean(np.abs(b - a)) / (np.mean((a + b) / 2) + 1e-9)
            print(f'  {LAYER_TITLES[c]:<24s} mean|delta| / mean|act| = {rel:.3f}')

        axes[0, 0].set_ylabel(base_name, fontsize=10, color=ink)
        axes[1, 0].set_ylabel(mix_name, fontsize=10, color=ink)
        axes[2, 0].set_ylabel('difference (mix − base)', fontsize=10, color=ink)
        for row, im, lab in ((0, im0, 'channel-mean |act| (layer-norm.)'),
                             (2, imd, 'signed Δ, same units')):
            cb = fig.colorbar(im, ax=axes[row, 1:].tolist(), fraction=0.012, pad=0.01)
            cb.set_label(lab, fontsize=8, color=ink)
            cb.ax.tick_params(labelsize=7, colors=ink)

        fig.suptitle(
            f'Encoder embedding maps — {pids[si]}  ({labels[si]}, slice {slcs[si]})  '
            f'— epoch-20 checkpoints, channel-mean |activation| per layer',
            fontsize=12, color='#222222')
        out_path = os.path.join(
            args.out_dir, f'embmap_{labels[si]}_{pids[si]}_slice{slcs[si]}.png')
        fig.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'saved {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
