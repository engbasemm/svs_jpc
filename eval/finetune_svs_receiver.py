"""
finetune_svs_receiver.py  -  receiver adaptation for SVS-JPC (Part C, "option 1"), and the same
adaptation for the HEVC + baseline pipeline as a fairness control.

FT_MODE=svs (default)
  The SVS-JPC sender (shared encoder, F3 projection, Multi_LDMIC codec) is frozen, so the bitstream
  and the rates are exactly those of the published checkpoints. A RECEIVER copy of the encoder, the
  cross-view fusion module and the two heads is fine-tuned on the reconstructions x_hat of the
  training frames.
  Output: <MTL_bitrates>/<svs run>_RXFT/models/weights_<FT_EPOCHS>/

FT_MODE=sender (option 3 on the sender side)
  Slim fusion (~6 M parameters instead of 351 M) and heads retrained on the original images with
  the sender encoder frozen, so the bitstream and the rates are unchanged.
  Output: <MTL_bitrates>/<svs run>_SXSLIM/models/weights_<FT_EPOCHS>/

FT_FUSION=slim (with FT_MODE=svs): the receiver gets the slim fusion instead of a copy of the
  351 M-parameter module (options 1 + 3). Output: <svs run>_RXFT_SLIM/...

FT_MODE=e2e (option 2: task-aware codec)
  The F3 projection and the Multi_LDMIC codec are trained together with the receiver, with the
  original loss lambda*255^2*MSE + R plus the receiver's detection and segmentation losses on x_hat
  (uncertainty-weighted, as in training). The sender encoder is frozen, so on-board perception is
  unchanged; the bitstream changes. The receiver starts from <run>_RXFT_SLIM if it exists.
  FT_LR_CODEC=1e-5 (codec), FT_LR (receiver). Output: <svs run>_E2E/models/weights_<FT_EPOCHS>/

FT_LATENT_SKIP=1 (FT_MODE=svs or e2e): the receiver also reads the transmitted latent y_hat (a projection of
  the sender's F3), added to its own F3 through a zero-initialised adapter. In e2e the task gradients then
  reach the projection through y_hat as well as through the pixels. Suffix _YSKIP.

FT_E2E_LAMBDA=0.3 (with FT_MODE=e2e): fine-tune the codec towards a new rate-distortion weight, starting from
  the checkpoint chosen by FT_LAMBDAS; higher values raise the rate. Output: <run>_E2E_L<value>.

FT_FUSION=geo / none: geometry-aware fusion (each camera attends to its two overlapping neighbours,
  with a viewing-direction encoding) or no fusion at all, for FT_MODE=svs (-> _RXFT_GEO / _RXFT_NOFUSION),
  sender (-> _SXGEO / _SXNOFUSION) and e2e (-> _E2E_GEO / _E2E_NOFUSION; warm start from _RXFT_<same>).

FT_MODE=hevc (fairness control)
  The training frames are coded with HM HEVC intra at each QP in FT_HEVC_QPS (cached as PNG), and a
  copy of the uncompressed baseline's encoder and heads is fine-tuned on the DECODED frames with the
  identical recipe. Without it, the comparison favours SVS-JPC, whose receiver is adapted to its own
  reconstructions while the HEVC rows run an unadapted baseline.
  Output: <MTL_bitrates>/<baseline run>_RXFT_HEVC_QP<qp>/models/weights_<FT_EPOCHS>/

Common recipe: BatchNorm statistics first re-estimated on the receiver's input (then fixed), the
trainer's own losses (YOLO detection loss, focal segmentation loss with ENet class weights,
uncertainty weighting), Adam, cosine decay, a fixed number of epochs, last epoch saved. Nothing is
selected on validation or test frames.

Run from the omnidet root (see run_svs_steps.sh):
  python eval/finetune_svs_receiver.py --config data/params.yaml

Settings (environment variables):
  FT_MODE=svs|hevc
  FT_LAMBDAS=0.08,0.05,0.03,0.01   (svs) which SVS-JPC models to adapt
  FT_HEVC_QPS=33,36,39,42          (hevc) which HEVC operating points to adapt the baseline to
  FT_EPOCHS=15  FT_LR=5e-5  FT_SEED=0
  FT_TRAIN_FUSION=1                 (svs) also fine-tune the fusion module (0: frozen)
  FT_AUG=1                          colour augmentation (svs: before coding, as in training;
                                    hevc: on the decoded frames, since the cache is fixed)
  FT_TRAIN_FILE=data/svs_bn_local.txt
"""
import os
import sys
import json
import time
import random
import datetime
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repository root
sys.path.insert(0, HERE)                   # eval/
import evaluate_mic_mtl_v2 as ev             # loaders, codec forward and BatchNorm helpers
from data_loader.synwoodscape_loader import SynWoodScapeRawDataset
from losses.detection_loss import ObjectDetectionLoss
from losses.semantic_loss import FocalLoss, CrossEntropyLoss2d
from losses.mtl_losses_regulated_attn import UncertaintyLoss
from models.detection_decoderORG import YoloDecoder
from models.semantic_decoder import SemanticDecoder
from main import collect_args
from utils import Tupperware

MODE = os.environ.get("FT_MODE", "svs")
assert MODE in ("svs", "hevc", "sender", "e2e", "joint"), MODE
LR_CODEC = float(os.environ.get("FT_LR_CODEC", "1e-5"))   # e2e: learning rate of the projection and codec
E2E_INIT = os.environ.get("FT_E2E_INIT", "")                 # e2e: receiver warm start (default _RXFT + fusion suffix)
FUSION = os.environ.get("FT_FUSION", "heavy")   # "heavy" (copy of the trained module), "slim", "geo" or "none"
assert FUSION in ("heavy", "slim", "geo", "none", "adapter"), FUSION
SUFFIX = {"heavy": "", "slim": "_SLIM", "geo": "_GEO", "none": "_NOFUSION", "adapter": "_ADAPTER"}
CODEC_GEO = ev._flag("FT_CODEC_GEO")                        # e2e: geometry-aware attention inside the LDMIC decoder
LR_GEO = float(os.environ.get("FT_LR_GEO", "1e-4"))        # e2e: learning rate of the new decoder blocks
LATENT_SKIP = ev._flag("FT_LATENT_SKIP")                    # svs / e2e: receiver also reads y_hat at F3
SKIP_SUFFIX = "_YSKIP" if LATENT_SKIP else ""
E2E_LAMBDA = os.environ.get("FT_E2E_LAMBDA", "")            # e2e: new rate-distortion weight (higher = more bits)
NEW_FUSION = FUSION if FUSION != "heavy" else "slim"   # sender and e2e runs always use a new fusion module
LAMBDAS = [s.strip() for s in os.environ.get("FT_LAMBDAS", "0.08,0.05,0.03,0.01").split(",") if s.strip()]
HEVC_QPS = [int(s) for s in os.environ.get("FT_HEVC_QPS", "33,36,39,42").split(",") if s.strip()]
EPOCHS = int(os.environ.get("FT_EPOCHS", "15"))
LR = float(os.environ.get("FT_LR", "5e-5"))
TRAIN_FUSION = ev._flag("FT_TRAIN_FUSION", True)
AUG = ev._flag("FT_AUG", True)
SEED = int(os.environ.get("FT_SEED", "0"))
SEM_WEIGHTS = dict(woodscape_enet=[3.25, 2.33, 20.42, 30.59, 38.4, 45.73, 10.76, 34.16, 44.3, 49.19],
                   woodscape_mfb=[0.04, 0.03, 0.43, 0.99, 2.02, 4.97, 0.17, 1.01, 3.32, 20.35])
JITTER = transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2),
                                hue=(-0.1, 0.1))   # the loader's training augmentation


def freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


def train_mode_frozen_bn(m):
    """Train mode, but every BatchNorm keeps its statistics (batch of four images is too small)."""
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()


def reset_bn(m, device):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.track_running_stats = True
            mod.running_mean = torch.zeros(mod.num_features, device=device)
            mod.running_var = torch.ones(mod.num_features, device=device)
            mod.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=device)
            mod.momentum = None


def train_targets(inputs, cams, device):
    """Detection labels as the trainer feeds them: normalised xywh, column 0 = image index."""
    parts = []
    for i, c in enumerate(cams):
        t = inputs.get(("detection_labels", 0, c))
        if t is not None and t.numel():
            t = t.view(-1, 6).clone()
            t[:, 0] = i
            parts.append(t)
    return torch.cat(parts, 0).to(device) if parts else None


class Indexed(Dataset):
    """(index, item) so that cached decoded frames can be matched to the sample."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return i, self.ds[i]


def make_indexed_loader(ds, args, shuffle):
    def collate(batch):
        return batch[0][0], ds.collate_fn([batch[0][1]])
    return DataLoader(Indexed(ds), batch_size=1, shuffle=shuffle, num_workers=args.num_workers,
                      pin_memory=True, drop_last=True, collate_fn=collate)


def train_receiver(rx, get_input, loader, cams, args, device, desc):
    """Fine-tune rx (enc, fusion or None, det, sem) on the images returned by get_input."""
    h, w = int(args.input_height), int(args.input_width)
    w_sem = torch.tensor(SEM_WEIGHTS[args.semantic_class_weighting], device=device)
    sem_crit = (FocalLoss(weight=w_sem, gamma=2, size_average=True) if args.semantic_loss == "focal_loss"
                else CrossEntropyLoss2d(weight=w_sem))
    det_crit = ObjectDetectionLoss(config=args)
    mtl = UncertaintyLoss(tasks="semantic_detection").to(device)
    params = [p for m in rx.values() if m is not None for p in m.parameters() if p.requires_grad]
    params += list(mtl.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(loader))
    n_train = sum(p.numel() for p in params) / 1e6
    log = []
    for epoch in range(EPOCHS):
        for k, m in rx.items():
            if m is not None:
                train_mode_frozen_bn(m)
        if rx.get("fusion") is not None and not any(p.requires_grad for p in rx["fusion"].parameters()):
            rx["fusion"].eval()
        run = {"detection_loss": 0.0, "semantic_loss": 0.0, "total": 0.0, "n": 0}
        for idx, inputs in tqdm(loader, desc=f"{desc} epoch {epoch + 1}/{EPOCHS}"):
            xin = get_input(idx, inputs)
            feats = ev.fuse(rx, ev.rx_encode(rx, xin))
            losses = {}
            det_out = rx["det"](feats, [w, h], train_targets(inputs, cams, device))
            losses["detection_loss"] = det_crit(det_out["yolo_output_dicts"],
                                                det_out["yolo_target_dicts"])["detection_loss"]
            labels = torch.cat([inputs[("semantic_labels", 0, 0, c)] for c in cams], 0).to(device)
            labels = labels.squeeze(1) if labels.dim() == 4 else labels
            losses["semantic_loss"] = sem_crit(rx["sem"](feats)[("semantic", 0)], labels)
            total = mtl(losses).mean()
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            run["detection_loss"] += losses["detection_loss"].item()
            run["semantic_loss"] += losses["semantic_loss"].item()
            run["total"] += total.item()
            run["n"] += 1
        row = {k: run[k] / max(1, run["n"]) for k in ("detection_loss", "semantic_loss", "total")}
        row["epoch"] = epoch + 1
        log.append(row)
        print(f"  epoch {epoch + 1}: det {row['detection_loss']:.4f}  sem {row['semantic_loss']:.4f}  "
              f"total {row['total']:.4f}")
    for m in rx.values():
        if m is not None:
            m.eval()
    return log, n_train


def save_receiver(rx, out, meta):
    os.makedirs(out, exist_ok=True)
    torch.save(rx["enc"].state_dict(), os.path.join(out, "encoder.pth"))
    if rx.get("fusion") is not None:
        torch.save(rx["fusion"].state_dict(), os.path.join(out, "feature_fusion.pth"))
    torch.save(rx["det"].state_dict(), os.path.join(out, "detection.pth"))
    torch.save(rx["sem"].state_dict(), os.path.join(out, "semantic.pth"))
    if rx.get("skip") is not None:
        torch.save(rx["skip"].state_dict(), os.path.join(out, "latent_skip.pth"))
        meta["latent_skip"] = True
    with open(os.path.join(out, "rxft_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved receiver to {out}  ({meta['minutes']:.1f} min)")


# --------------------------------------------------------------------------------------
# FT_MODE=svs
# --------------------------------------------------------------------------------------
def run_svs(args, device, train_file):
    ds = SynWoodScapeRawDataset(data_path=args.dataset_dir, path_file=train_file, is_train=AUG, config=args)
    cams = ev.CAMS = list(ds.all_cam_sides)
    loader = make_indexed_loader(ds, args, shuffle=True)
    h, w = int(args.input_height), int(args.input_width)
    img_key = "color_aug" if AUG else "color"
    for cfg in [c for c in ev.DL_MODEL_CONFIGS if c["kind"] == "svs" and str(c["lam"]) in LAMBDAS]:
        t0 = time.time()
        print(f"\n=== {cfg['name']} ===")
        b = ev.load_bundle(cfg, args, device)
        if b is None:
            print(f"[skip] not found: {ev.run_dir_of(cfg, args)}")
            continue
        for k in ("enc", "proj", "comp"):
            freeze(b[k])                                   # sender: bitstream unchanged
        enc_rx, bn_msg = ev.rx_bn_from_xhat(b, args, device, ev.BN_RECALIB_N or 250, SEED)
        rx = dict(enc=enc_rx, **ev.clone_heads(b, args, device))
        if FUSION != "heavy":                               # slim / geo start as identity; none = no fusion
            rx["fusion"] = ev.make_fusion(FUSION, b["enc"].num_ch_enc[-1], device)
        if LATENT_SKIP:
            rx["skip"] = ev.LatentSkip(c_out=b["enc"].num_ch_enc[-2]).to(device)
        del b["fusion"], b["det"], b["sem"]
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for k, m in rx.items():
            for p in (m.parameters() if m is not None else []):
                p.requires_grad_(k != "fusion" or TRAIN_FUSION)
        print(f"  receiver BatchNorm: {bn_msg}; {FUSION} fusion {'trained' if TRAIN_FUSION else 'frozen'}")

        def get_input(idx, inputs):
            x4 = ev.to_8bit(torch.cat([inputs[(img_key, 0, 0, c)] for c in cams], 0).to(device))
            with torch.no_grad():
                _, x_hat, _ = ev.compress(b, x4, h, w)
            return x_hat if ev.PERCEPTION_INPUT == "float" else ev.to_8bit(x_hat)

        log, n_train = train_receiver(rx, get_input, loader, cams, args, device, cfg["name"])
        meta = dict(mode="svs", fusion=FUSION, sender_checkpoint=ev.run_dir_of(cfg, args), lambda_=cfg["lam"], epochs=EPOCHS,
                    lr=LR, train_fusion=TRAIN_FUSION, augmentation=AUG, seed=SEED, train_file=train_file,
                    n_train_tuples=len(ds), receiver_bn_init=bn_msg, trainable_params_M=n_train,
                    perception_input=ev.PERCEPTION_INPUT, log=log, minutes=(time.time() - t0) / 60,
                    date=datetime.datetime.now().isoformat(timespec="seconds"),
                    device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
        suffix = "_RXFT" + SUFFIX[FUSION] + SKIP_SUFFIX
        save_receiver(rx, os.path.join(args.pretrained_weights, cfg["pretrained_path_segment"] + suffix,
                                       "models", f"weights_{EPOCHS}"), meta)
        del b, rx
        if device.type == "cuda":
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# FT_MODE=e2e  (option 2: task-aware codec)
# --------------------------------------------------------------------------------------
def run_e2e(args, device, train_file):
    """Codec (F3 projection, entropy model, joint decoder) and receiver trained together, with the
    original rate-distortion loss plus the detection and segmentation losses of the receiver that
    perceives the reconstructions. The sender encoder stays frozen, so on-board perception is
    unchanged; the bitstream changes, so rates must be re-measured. Output: <run>_E2E."""
    import math
    import torch.nn.functional as F
    ds = SynWoodScapeRawDataset(data_path=args.dataset_dir, path_file=train_file, is_train=AUG, config=args)
    cams = ev.CAMS = list(ds.all_cam_sides)
    loader = make_indexed_loader(ds, args, shuffle=True)
    h, w = int(args.input_height), int(args.input_width)
    img_key = "color_aug" if AUG else "color"
    w_sem = torch.tensor(SEM_WEIGHTS[args.semantic_class_weighting], device=device)
    sem_crit = (FocalLoss(weight=w_sem, gamma=2, size_average=True) if args.semantic_loss == "focal_loss"
                else CrossEntropyLoss2d(weight=w_sem))
    det_crit = ObjectDetectionLoss(config=args)
    for cfg in [c for c in ev.DL_MODEL_CONFIGS if c["kind"] == "svs" and str(c["lam"]) in LAMBDAS]:
        t0 = time.time()
        lam = float(E2E_LAMBDA or cfg["lam"])
        print(f"\n=== {cfg['name']} (end-to-end task-aware codec) ===")
        b = ev.load_bundle(cfg, args, device)
        if b is None:
            print(f"[skip] not found: {ev.run_dir_of(cfg, args)}")
            continue
        freeze(b["enc"])                                   # sender encoder: on-board perception unchanged
        # receiver: warm start from an earlier receiver fine-tune if one exists
        init_dir = ev.latest_weights_dir(os.path.join(args.pretrained_weights,
                                                      cfg["pretrained_path_segment"] + (E2E_INIT or "_RXFT" + SUFFIX[NEW_FUSION] + SKIP_SUFFIX),
                                                      "models"))
        if init_dir:
            rx, kind = ev.load_ft_dir(init_dir, b, args, device)
            init_msg = f"receiver warm start from {init_dir} ({kind} fusion)"
        else:
            enc_rx, bn_msg = ev.rx_bn_from_xhat(b, args, device, ev.BN_RECALIB_N or 250, SEED)
            rx = dict(enc=enc_rx, **ev.clone_heads(b, args, device))
            rx["fusion"] = ev.make_fusion(NEW_FUSION, b["enc"].num_ch_enc[-1], device)
            kind, init_msg = NEW_FUSION, f"receiver from scratch: {bn_msg}, {NEW_FUSION} fusion"
        if LATENT_SKIP and rx.get("skip") is None:
            rx["skip"] = ev.LatentSkip(c_out=b["enc"].num_ch_enc[-2]).to(device)
            init_msg += "; new latent skip"
        del b["fusion"], b["det"], b["sem"]
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {init_msg}")
        if CODEC_GEO:
            ev.add_codec_geo(b["comp"], cams)
        named = list(b["proj"].named_parameters()) + list(b["comp"].named_parameters())
        geo_params = [p for n, p in named if n.startswith("geo_")]
        codec_params = [p for n, p in named if "quantiles" not in n and not n.startswith("geo_")]
        for p in codec_params + geo_params:
            p.requires_grad_(True)
        rx_params = [p for m in rx.values() if m is not None for p in m.parameters()]
        for p in rx_params:
            p.requires_grad_(True)
        mtl = UncertaintyLoss(tasks="compression_semantic_detection", enable_compression=True).to(device)
        opt = torch.optim.Adam([{"params": rx_params + list(mtl.parameters()), "lr": LR},
                                {"params": codec_params, "lr": LR_CODEC}]
                               + ([{"params": geo_params, "lr": LR_GEO}] if geo_params else []))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(loader))
        all_params = rx_params + codec_params + geo_params + list(mtl.parameters())
        log = []
        for epoch in range(EPOCHS):
            for m in rx.values():
                if m is not None:
                    train_mode_frozen_bn(m)
            b["proj"].train()
            b["comp"].train()                               # noisy likelihoods, STE-rounded y for the decoder (as trained)
            run = {k: 0.0 for k in ("detection_loss", "semantic_loss", "compression_loss", "bpp", "mse", "total")}
            n = 0
            for idx, inputs in tqdm(loader, desc=f"{cfg['name']} e2e epoch {epoch + 1}/{EPOCHS}"):
                x4 = ev.to_8bit(torch.cat([inputs[(img_key, 0, 0, c)] for c in cams], 0).to(device))
                with torch.no_grad():
                    f = b["enc"](x4)
                out = b["comp"](b["proj"](f[-2]), num_camera=len(cams))
                bpp = sum(torch.log(l).sum() for l in out["likelihoods"].values()) / (-math.log(2) * len(cams) * h * w)
                mse = F.mse_loss(out["x_hat"], x4)
                losses = {"compression_loss": lam * 255 ** 2 * mse + bpp}
                xin = out["x_hat"] if ev.PERCEPTION_INPUT == "float" else ev.to_8bit(out["x_hat"])
                feats = ev.fuse(rx, ev.rx_encode(rx, xin))
                det_out = rx["det"](feats, [w, h], train_targets(inputs, cams, device))
                losses["detection_loss"] = det_crit(det_out["yolo_output_dicts"],
                                                    det_out["yolo_target_dicts"])["detection_loss"]
                labels = torch.cat([inputs[("semantic_labels", 0, 0, c)] for c in cams], 0).to(device)
                labels = labels.squeeze(1) if labels.dim() == 4 else labels
                losses["semantic_loss"] = sem_crit(rx["sem"](feats)[("semantic", 0)], labels)
                comp_part, det_part, sem_part = (losses["compression_loss"].item(), losses["detection_loss"].item(),
                                                 losses["semantic_loss"].item())
                total = mtl(losses).mean()
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 10.0)
                opt.step()
                sched.step()
                run["detection_loss"] += det_part
                run["semantic_loss"] += sem_part
                run["compression_loss"] += comp_part
                run["bpp"] += bpp.item()
                run["mse"] += mse.item()
                run["total"] += total.item()
                n += 1
            row = {k: v / max(1, n) for k, v in run.items()}
            row["epoch"] = epoch + 1
            log.append(row)
            print(f"  epoch {epoch + 1}: det {row['detection_loss']:.4f}  sem {row['semantic_loss']:.4f}  "
                  f"bpp {row['bpp']:.4f}  psnr {-10 * math.log10(max(row['mse'], 1e-10)):.2f}  total {row['total']:.4f}")
        for m in list(rx.values()) + [b["proj"], b["comp"]]:
            if m is not None:
                m.eval()
        out_dir = os.path.join(args.pretrained_weights,
                               cfg["pretrained_path_segment"] + "_E2E" + ("" if kind == "slim" else SUFFIX[kind])
                               + ("_CGEO" if CODEC_GEO else "") + SKIP_SUFFIX + (f"_L{E2E_LAMBDA}" if E2E_LAMBDA else ""),
                               "models",
                               f"weights_{EPOCHS}")
        meta = dict(mode="e2e", fusion=kind, codec_geo=CODEC_GEO, lr_geo=LR_GEO,
                    geo_params_M=sum(p.numel() for p in geo_params) / 1e6, sender_checkpoint=ev.run_dir_of(cfg, args), lambda_=lam, epochs=EPOCHS,
                    lr=LR, lr_codec=LR_CODEC, augmentation=AUG, seed=SEED, train_file=train_file,
                    n_train_tuples=len(ds), receiver_init=init_msg,
                    sender="encoder frozen; projection and codec trained (bitstream changed)",
                    trainable_params_M=sum(p.numel() for p in all_params) / 1e6, log=log,
                    minutes=(time.time() - t0) / 60, date=datetime.datetime.now().isoformat(timespec="seconds"),
                    device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
        save_receiver(rx, out_dir, meta)
        torch.save(b["proj"].state_dict(), os.path.join(out_dir, "compression_projection.pth"))
        torch.save(b["comp"].state_dict(), os.path.join(out_dir, "decoder.pth"))
        del b, rx, opt, mtl
        if device.type == "cuda":
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# FT_MODE=sender  (option 3 on the sender side)
# --------------------------------------------------------------------------------------
def run_sender(args, device, train_file):
    """Slim fusion and heads retrained on the ORIGINAL images with the sender encoder frozen, so the
    bitstream and the rates stay those of the checkpoint. Output: <run>_SXSLIM."""
    ds = SynWoodScapeRawDataset(data_path=args.dataset_dir, path_file=train_file, is_train=AUG, config=args)
    cams = ev.CAMS = list(ds.all_cam_sides)
    loader = make_indexed_loader(ds, args, shuffle=True)
    img_key = "color_aug" if AUG else "color"
    for cfg in [c for c in ev.DL_MODEL_CONFIGS if c["kind"] == "svs" and str(c["lam"]) in LAMBDAS]:
        t0 = time.time()
        print(f"\n=== {cfg['name']} (sender, {NEW_FUSION} fusion) ===")
        b = ev.load_bundle(cfg, args, device)
        if b is None:
            print(f"[skip] not found: {ev.run_dir_of(cfg, args)}")
            continue
        for k in ("enc", "proj", "comp"):
            freeze(b[k])
        heads = ev.clone_heads(b, args, device)
        tx = dict(enc=b["enc"], fusion=ev.make_fusion(NEW_FUSION, b["enc"].num_ch_enc[-1], device),
                  det=heads["det"], sem=heads["sem"])
        del b["fusion"], b["det"], b["sem"], heads
        if device.type == "cuda":
            torch.cuda.empty_cache()

        def get_input(idx, inputs):
            return ev.to_8bit(torch.cat([inputs[(img_key, 0, 0, c)] for c in cams], 0).to(device))

        log, n_train = train_receiver(tx, get_input, loader, cams, args, device, cfg["name"] + " sender")
        meta = dict(mode="sender", fusion=NEW_FUSION, sender_checkpoint=ev.run_dir_of(cfg, args), lambda_=cfg["lam"],
                    epochs=EPOCHS, lr=LR, augmentation=AUG, seed=SEED, train_file=train_file,
                    n_train_tuples=len(ds), encoder="sender encoder frozen (bitstream unchanged)",
                    trainable_params_M=n_train, log=log, minutes=(time.time() - t0) / 60,
                    date=datetime.datetime.now().isoformat(timespec="seconds"),
                    device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
        save_receiver(tx, os.path.join(args.pretrained_weights, cfg["pretrained_path_segment"] + "_SX" + (SUFFIX[NEW_FUSION][1:]),
                                       "models", f"weights_{EPOCHS}"), meta)
        del b, tx
        if device.type == "cuda":
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# FT_MODE=hevc  (fairness control)
# --------------------------------------------------------------------------------------
def _hevc_frame_task(task):
    """One frame: four camera images (uint8 HWC) x all QPs -> cached decoded PNGs and bpp."""
    torch.set_num_threads(1)
    arrays, cams, frame, qps, hm, cache_root = task
    out = {}
    for qp in qps:
        bpps = []
        for c, a in zip(cams, arrays):
            png = os.path.join(cache_root, f"QP{qp}", f"{frame}_{c}.png")
            meta = png + ".bpp"
            if os.path.exists(png) and os.path.exists(meta):
                bpps.append(float(open(meta).read()))
                continue
            orig = torch.from_numpy(a).permute(2, 0, 1).float() / 255.0
            r = ev.compress_hevc(orig, qp, hm, png)
            with open(meta, "w") as f:
                f.write(str(r["BPP"]))
            bpps.append(r["BPP"])
        out[qp] = float(np.nanmean(bpps))
    return frame, out


def hevc_cache(args, ds, cams, qps, hm, cache_root):
    """Code every training frame with HM at each QP once; later runs reuse the cache."""
    frames = list(ds.unique_frame_indices)

    def tasks():
        for i in range(len(ds)):
            it = ds.collate_fn([ds[i]])
            arrays = [(ev.to_8bit(it[("color", 0, 0, c)])[0].permute(1, 2, 0).numpy() * 255).round()
                      .astype(np.uint8) for c in cams]
            yield arrays, cams, frames[i], qps, hm, cache_root

    rates = {qp: [] for qp in qps}
    n_workers = int(os.environ.get("EVAL_CODEC_WORKERS", max(1, (os.cpu_count() or 2) // 2)))
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for frame, r in tqdm(pool.imap_unordered(_hevc_frame_task, tasks(), chunksize=2), total=len(ds),
                             desc="HEVC coding of training frames"):
            for qp, v in r.items():
                rates[qp].append(v)
    return {qp: float(np.nanmean(v)) for qp, v in rates.items()}


def run_hevc(args, device, train_file):
    hm_base = os.path.abspath(os.path.join(HERE, "..", ".."))
    hm = (os.path.join(hm_base, "HM", "cfg", "encoder_intra_main.cfg"),
          os.path.join(hm_base, "HM", "cfg", "per-sequence", "Traffic_Fisheye8K.cfg"),
          os.path.join(hm_base, "HM", "bin", "TAppEncoderStatic"),
          os.path.join(hm_base, "HM", "bin", "TAppDecoderStatic"))
    if not os.path.exists(hm[2]):
        sys.exit(f"HM encoder not found: {hm[2]}")
    ds = SynWoodScapeRawDataset(data_path=args.dataset_dir, path_file=train_file, is_train=False, config=args)
    cams = ev.CAMS = list(ds.all_cam_sides)
    frames = list(ds.unique_frame_indices)
    cache_root = os.path.join(args.output_directory, "hevc_train_cache")
    rates = hevc_cache(args, ds, cams, HEVC_QPS, hm, cache_root)
    print("  mean bpp of the coded training frames:", {f"QP{q}": round(v, 4) for q, v in rates.items()})
    loader = make_indexed_loader(ds, args, shuffle=True)
    base_cfg = ev.DL_MODEL_CONFIGS[0]
    to_t = transforms.ToTensor()

    for qp in HEVC_QPS:
        t0 = time.time()
        print(f"\n=== baseline adapted to HEVC QP{qp} ===")
        b = ev.load_bundle(base_cfg, args, device)

        def decoded(idx, aug):
            xs = []
            for c in cams:
                img = Image.open(os.path.join(cache_root, f"QP{qp}", f"{frames[idx]}_{c}.png")).convert("RGB")
                if aug and random.random() > 0.5:
                    img = JITTER(img)
                xs.append(to_t(img))
            return torch.stack(xs).to(device)

        # BatchNorm statistics of the receiver copy from decoded training frames (then fixed)
        import copy
        enc_rx = copy.deepcopy(b["enc"])
        reset_bn(enc_rx, device)
        g = torch.Generator().manual_seed(SEED)
        idx_bn = torch.randperm(len(ds), generator=g)[:(ev.BN_RECALIB_N or 250)].tolist()
        enc_rx.train()
        with torch.no_grad():
            for i in tqdm(idx_bn, desc=f"receiver BN from HEVC QP{qp}"):
                enc_rx(decoded(i, False))
        enc_rx.eval()
        bn_msg = f"receiver copy, statistics from HEVC QP{qp} decoded frames of {len(idx_bn)} training tuples"
        det = YoloDecoder(b["enc"].num_ch_enc, args).to(device)
        det.load_state_dict(b["det"].state_dict())
        sem = SemanticDecoder(b["enc"].num_ch_enc, n_classes=args.semantic_num_classes).to(device)
        sem.load_state_dict(b["sem"].state_dict())
        rx = dict(enc=enc_rx, fusion=None, det=det, sem=sem)
        for m in (enc_rx, det, sem):
            for p in m.parameters():
                p.requires_grad_(True)
        print(f"  {bn_msg}")
        log, n_train = train_receiver(rx, lambda idx, inputs: decoded(idx, AUG), loader, cams, args, device,
                                      f"baseline@HEVC QP{qp}")
        meta = dict(mode="hevc", qp=qp, train_bpp=rates[qp], base_checkpoint=ev.run_dir_of(base_cfg, args),
                    epochs=EPOCHS, lr=LR, augmentation=f"{AUG} (applied to decoded frames)", seed=SEED,
                    train_file=train_file, n_train_tuples=len(ds), receiver_bn_init=bn_msg,
                    trainable_params_M=n_train, log=log, minutes=(time.time() - t0) / 60,
                    date=datetime.datetime.now().isoformat(timespec="seconds"),
                    device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
        save_receiver(rx, os.path.join(args.pretrained_weights, f"{base_cfg['pretrained_path_segment']}_RXFT_HEVC_QP{qp}",
                                       "models", f"weights_{EPOCHS}"), meta)
        del b, rx
        if device.type == "cuda":
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# FT_MODE=joint  (sender-side multi-task training, one network on one SoC)
# --------------------------------------------------------------------------------------
JOINT_LAMBDA = os.environ.get("FT_JOINT_LAMBDA", "")        # target rate-distortion weight (default: source's)
JOINT_NOCODEC = ev._flag("FT_JOINT_NOCODEC")                # perception-only control: same schedule, no codec
JOINT_TAG = os.environ.get("FT_JOINT_TAG", "")
LR_ENC = float(os.environ.get("FT_LR_ENC", "2e-5"))         # encoder and heads (already trained)
LR_NEW = float(os.environ.get("FT_LR_NEW", "1e-4"))         # projection, codec, new fusion / geometry blocks
JOINT_INIT = os.environ.get("FT_JOINT_INIT", "")            # warm start from JOINT/<tag> (fusion and codec as saved)
SIDE = ev._flag("FT_SIDE")                                  # add the side analysis branch to the codec latent
DETACH_CODEC = ev._flag("FT_DETACH_CODEC")                  # codec loss does not reach the shared encoder


def run_joint(args, device, train_file):
    """All sender modules trained together, as in the original multi-task training: shared encoder,
    fusion (FT_FUSION: heavy = released module, slim, geo, none), detection and segmentation heads, and
    (unless FT_JOINT_NOCODEC) the F3 projection and LDMIC codec (FT_CODEC_GEO adds the geometry-aware
    decoder blocks). Loss: uncertainty-weighted detection, segmentation and lambda*255^2*MSE + rate,
    with lambda = FT_JOINT_LAMBDA. Perception is on the encoder features of the original images; no
    second pass. Starts from the checkpoint chosen by FT_LAMBDAS (or the baseline with NOCODEC).
    Output: <MTL_bitrates>/JOINT/<FT_JOINT_TAG>/models/weights_<FT_EPOCHS>/"""
    import math
    import torch.nn.functional as F
    ds = SynWoodScapeRawDataset(data_path=args.dataset_dir, path_file=train_file, is_train=AUG, config=args)
    cams = ev.CAMS = list(ds.all_cam_sides)
    loader = make_indexed_loader(ds, args, shuffle=True)
    h, w = int(args.input_height), int(args.input_width)
    img_key = "color_aug" if AUG else "color"
    w_sem = torch.tensor(SEM_WEIGHTS[args.semantic_class_weighting], device=device)
    sem_crit = (FocalLoss(weight=w_sem, gamma=2, size_average=True) if args.semantic_loss == "focal_loss"
                else CrossEntropyLoss2d(weight=w_sem))
    det_crit = ObjectDetectionLoss(config=args)
    if JOINT_INIT:
        cfg = dict(name=f"JOINT_{JOINT_INIT}", kind="joint", lam=None, tag=JOINT_INIT)
    elif JOINT_NOCODEC:
        cfg = ev.DL_MODEL_CONFIGS[0]
    else:
        cfgs = [c for c in ev.DL_MODEL_CONFIGS if c["kind"] == "svs" and str(c["lam"]) in LAMBDAS]
        assert len(cfgs) == 1, f"FT_LAMBDAS must name one source checkpoint for joint mode: {LAMBDAS}"
        cfg = cfgs[0]
    if JOINT_INIT:
        b = ev.load_bundle(cfg, args, device)
        if b is None:
            sys.exit(f"warm-start run not found: JOINT/{JOINT_INIT}")
    lam = float(JOINT_LAMBDA or (cfg["lam"] or 0))
    tag = JOINT_TAG or ("BASE_FT" if JOINT_NOCODEC else f"L{lam}_{FUSION}" + ("_CGEO" if CODEC_GEO else ""))
    out_dir = os.path.join(args.pretrained_weights, "JOINT", tag, "models", f"weights_{EPOCHS}")
    if os.path.exists(os.path.join(out_dir, "joint_meta.json")):
        print(f"[skip] {tag}: already trained ({out_dir})")
        return
    t0 = time.time()
    if JOINT_INIT:                                       # everything as saved by that run
        fusion = b["fusion"]
        fusion_kind = b["joint_meta"]["fusion"]
        codec_geo = bool(b["joint_meta"].get("codec_geo"))
        nocodec = bool(b["joint_meta"].get("nocodec"))
        source = b["run_dir"]
    else:
        b = ev.load_bundle(cfg, args, device)
        if b is None:
            sys.exit(f"source checkpoint not found: {ev.run_dir_of(cfg, args)}")
        fusion = b["fusion"] if FUSION == "heavy" else ev.make_fusion(FUSION, b["enc"].num_ch_enc[-1], device)
        fusion_kind, codec_geo, nocodec, source = FUSION, CODEC_GEO, JOINT_NOCODEC, ev.run_dir_of(cfg, args)
        if JOINT_NOCODEC:
            b["proj"] = b["comp"] = None
        elif CODEC_GEO:
            ev.add_codec_geo(b["comp"], cams)
    print(f"\n=== joint {tag}: from {cfg['name']}, lambda {lam}, fusion {fusion_kind}, codec geo {codec_geo}, "
          f"side {SIDE or b.get('side') is not None}, codec loss {'stops at F3' if DETACH_CODEC else 'reaches encoder'}, "
          f"{EPOCHS} epochs ===")
    side = b.get("side")
    if SIDE and side is None and not nocodec:
        side = ev.SideAnalysis().to(device)
    m = dict(enc=b["enc"], fusion=fusion, det=b["det"], sem=b["sem"], proj=b["proj"], comp=b["comp"], side=side)
    old, new = [], []
    for k, mod in m.items():
        if mod is None:
            continue
        for n, p in mod.named_parameters():
            if "quantiles" in n:
                continue
            p.requires_grad_(True)
            (new if k in ("proj", "comp", "side") or n.startswith("geo_") or (k == "fusion" and fusion_kind != "heavy")
             else old).append(p)
    tasks = "semantic_detection" if nocodec else "compression_semantic_detection"
    mtl = UncertaintyLoss(tasks=tasks, enable_compression=not nocodec).to(device)
    groups = [{"params": old, "lr": LR_ENC}, {"params": new + list(mtl.parameters()), "lr": LR_NEW}]
    opt = torch.optim.Adam([g for g in groups if g["params"]])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(loader))
    all_params = old + new + list(mtl.parameters())
    tx = dict(enc=m["enc"], fusion=m["fusion"], det=m["det"], sem=m["sem"])
    log = []
    for epoch in range(EPOCHS):
        for mod in m.values():
            if mod is not None:
                train_mode_frozen_bn(mod)
        run = {k: 0.0 for k in ("detection_loss", "semantic_loss", "compression_loss", "bpp", "mse", "total")}
        n = 0
        for idx, inputs in tqdm(loader, desc=f"joint {tag} epoch {epoch + 1}/{EPOCHS}"):
            x4 = ev.to_8bit(torch.cat([inputs[(img_key, 0, 0, c)] for c in cams], 0).to(device))
            f = m["enc"](x4)
            feats = ev.fuse(tx, f)
            losses = {}
            det_out = m["det"](feats, [w, h], train_targets(inputs, cams, device))
            losses["detection_loss"] = det_crit(det_out["yolo_output_dicts"],
                                                det_out["yolo_target_dicts"])["detection_loss"]
            labels = torch.cat([inputs[("semantic_labels", 0, 0, c)] for c in cams], 0).to(device)
            labels = labels.squeeze(1) if labels.dim() == 4 else labels
            losses["semantic_loss"] = sem_crit(m["sem"](feats)[("semantic", 0)], labels)
            if not nocodec:
                f3 = f[-2].detach() if DETACH_CODEC else f[-2]
                out = m["comp"](ev.project(m, f3, x4), num_camera=len(cams))
                bpp = sum(torch.log(l).sum() for l in out["likelihoods"].values()) / (-math.log(2) * len(cams) * h * w)
                mse = F.mse_loss(out["x_hat"], x4)
                losses["compression_loss"] = lam * 255 ** 2 * mse + bpp
                run["bpp"] += bpp.item()
                run["mse"] += mse.item()
                run["compression_loss"] += losses["compression_loss"].item()
            run["detection_loss"] += losses["detection_loss"].item()
            run["semantic_loss"] += losses["semantic_loss"].item()
            total = mtl(losses).mean()
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 10.0)
            opt.step()
            sched.step()
            run["total"] += total.item()
            n += 1
        row = {k: v / max(1, n) for k, v in run.items()}
        row["epoch"] = epoch + 1
        log.append(row)
        extra = "" if nocodec else (f"  bpp {row['bpp']:.4f}  psnr "
                                         f"{-10 * math.log10(max(row['mse'], 1e-10)):.2f}")
        print(f"  epoch {epoch + 1}: det {row['detection_loss']:.4f}  sem {row['semantic_loss']:.4f}{extra}  "
              f"total {row['total']:.4f}   [{(time.time() - t0) / 60:.0f} min]")
    os.makedirs(out_dir, exist_ok=True)
    for k, fn in (("enc", "encoder.pth"), ("fusion", "feature_fusion.pth"), ("det", "detection.pth"),
                  ("sem", "semantic.pth"), ("proj", "compression_projection.pth"), ("comp", "decoder.pth"),
                  ("side", "side_analysis.pth")):
        if m[k] is not None:
            torch.save(m[k].state_dict(), os.path.join(out_dir, fn))
    meta = dict(mode="joint", tag=tag, source=source, source_name=cfg["name"], lambda_=lam,
                fusion=fusion_kind if m["fusion"] is not None else "none", codec_geo=codec_geo and not nocodec,
                nocodec=nocodec, side=m["side"] is not None, detach_codec=DETACH_CODEC, init=JOINT_INIT or None, epochs=EPOCHS, lr_enc=LR_ENC, lr_new=LR_NEW, augmentation=AUG, seed=SEED,
                train_file=train_file, n_train_tuples=len(ds), log=log, minutes=(time.time() - t0) / 60,
                trainable_params_M=sum(p.numel() for p in all_params) / 1e6,
                date=datetime.datetime.now().isoformat(timespec="seconds"),
                device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    with open(os.path.join(out_dir, "joint_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  saved {out_dir}  ({meta['minutes']:.1f} min)")


def main():
    config = collect_args()
    args = Tupperware(yaml.safe_load(open(config.config)))
    args.dataset = "synwoodscape_raw"
    args.train = "compression_semantic_detection"
    args.batch_size = 1
    args.frame_idxs = [0]
    args.has_detection_annotations = True
    args.has_semantic_annotations = True
    args.pretrained_weights = os.environ.get("EVAL_WEIGHTS", ev.DEFAULT_WEIGHTS)
    if not os.path.isdir(args.pretrained_weights):
        sys.exit(f"Checkpoint folder not found: {args.pretrained_weights}")
    train_file = os.environ.get("FT_TRAIN_FILE", "data/svs_bn_local.txt")
    ev.check_lists(args, {"fine-tuning": train_file})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = str(device) if device.type == "cpu" else args.device
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    {"svs": run_svs, "hevc": run_hevc, "sender": run_sender, "e2e": run_e2e, "joint": run_joint}[MODE](
        args, device, train_file)


if __name__ == "__main__":
    main()
