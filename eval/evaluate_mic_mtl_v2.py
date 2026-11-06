"""
evaluate_mic_mtl_v2.py  -  corrected Part C (SVS-JPC) evaluation.

Place it in eval/ of the SVS_MTL code, next to evaluate_mic_mtl.py, and run it from the
repository root so the imports resolve:
    python eval/evaluate_mic_mtl_v2.py --config data/params.yaml

What it fixes relative to evaluate_mic_mtl.py (same numbering style as evaluate_mtl_v4.py)
  [C1]  Two perception paths for every SVS-JPC model, from one forward pass:
          sender   (S): heads on the encoder features of the original images, F4 fused
                         (what the released script and training measure);
          receiver (R): heads on the encoder features of the reconstructions x_hat, F4 fused
                         (what a receiver can compute from the bitstream; Part B protocol).
        Written as two result rows, <model>__S and <model>__R.
  [C2]  BatchNorm runs as each checkpoint was trained (running statistics if the checkpoint has
        them). Encoders without statistics get them from training tuples (forward passes only,
        no weight update), as in evaluate_mtl_v4.py. The old script forced per-image
        statistics on every encoder, the baseline included.
  [C3]  Every module loads strictly (encoder, fusion, projection, codec, heads). A missing or
        unexpected key stops the run; the old script caught the error and went on with random
        weights.
  [C4]  Codec rows get perception: each camera image is coded with HM HEVC intra and JPEG2000,
        decoded, and the four decoded images go through the uncompressed baseline. The old
        script reported rate and PSNR only. Codec rates are file bits / (H x W) per camera image,
        averaged over the four cameras: the same basis as the learned rate.
  [C5]  Rates: entropy estimate of y and z per pixel of one camera image, averaged over cameras
        (as the training loss), plus the per-camera rates.
  [C6]  The 'fusion-only' entry of the old script loaded the baseline weights with fusion off,
        i.e. it was the baseline twice. It is dropped.
  [C7]  Only SynWoodScape is evaluated (true synchronised tuples). The old script also ran a
        'virtual multi-view' WoodScape set built from random unrelated images, on the TRAINING
        list. That is removed.
  [C8]  Predictions on images without ground-truth boxes count as false positives; mIoU is given
        with and without the void class; 8-bit inputs and reconstructions for PSNR / MS-SSIM.
  [C9]  Per-tuple records (key = frame index, the four cameras together) are saved in the format
        stats_tests.py reads, so paired bootstrap intervals and paired tests work unchanged:
            python eval/stats_tests.py <run>/SynWoodScape_val/per_image --ref SVS_Baseline_no_compression
        The unit of resampling is the four-camera tuple, because the cameras of one frame are
        correlated.
  [C10] run_meta.json records platform, GPU, flags, checkpoints and lists.
  [C11] Before any model runs, every listed frame is checked for all four cameras.
  [C12] Codec workers are started with 'spawn' and one thread each (fork after CUDA deadlocked).
  [C13] EVAL_RX_BN=xhat: the receiver-side encoder pass uses its own BatchNorm statistics,
        estimated from reconstructions of training tuples (Part B protocol); the sender is unchanged.
  [C14] EVAL_RX_FT=1: the receiver path uses the encoder, fusion and heads fine-tuned on x_hat by
        finetune_svs_receiver.py (sender, bitstream and rates unchanged).
  [C15] EVAL_CODEC_FT=1: HEVC rows also with the baseline fine-tuned on HEVC-decoded training frames
        at the same QP (fairness control). EVAL_CODECS=hevc skips JPEG2000.
  [C16] Slim fusion (1x1 to 256 channels, joint context transfer, 1x1 back, residual; ~6 M parameters):
        EVAL_RX_FT_SUFFIX=_RXFT_SLIM loads the slim fine-tuned receiver; EVAL_SX_SLIM=1 uses the slim
        sender-side fusion and heads (sender encoder frozen, bitstream unchanged).
        EVAL_CODEC_WORKERS=1 runs the codecs in the main process.
  [C17] EVAL_E2E=1: task-aware codec (projection, codec and slim receiver trained end to end with the
        perception losses; finetune_svs_receiver.py FT_MODE=e2e). Sender encoder unchanged; the bitstream
        changes, so the rate and PSNR are re-measured.
  [C18] Geometry-aware fusion (GeoFusion, FT_FUSION=geo): each camera attends only to its two overlapping
        neighbours, with a Fourier encoding of each position's viewing direction in the vehicle frame
        (nominal equidistant fisheye, nominal yaw + learned offset). FT_FUSION=none: no fusion (control).
        EVAL_SX_SUFFIX / EVAL_E2E_SUFFIX select the sender and end-to-end runs to evaluate.
  [C23] Side analysis branch (FT_SIDE=1): a light image analysis transform added to the F3 projection,
        so the latent can carry pixel detail; FT_DETACH_CODEC=1 stops the codec loss at F3.
  [C24] AdapterFusion (FT_FUSION=adapter): slim-fusion-sized per-camera adapter, no cross-view exchange.
  [C22] Sender-side evaluation of jointly trained runs (FT_MODE=joint): EVAL_JOINT_RUNS, EVAL_SENDER_ONLY,
        EVAL_CODEC_PERC_JOINT, EVAL_NO_RELEASED; EVAL_HEVC_QPS sets the HEVC QPs (e.g. 24,...,45 for 0.1-1 bpp).
  [C20] FT_LATENT_SKIP=1: the receiver also reads the transmitted latent y_hat, added to its F3 through a
        zero-initialised adapter (LatentSkip); loaded automatically from the run's rxft_meta.json.
  [C19] FT_CODEC_GEO=1 (with FT_MODE=e2e): geometry-aware attention also inside the LDMIC joint decoder
        (add_codec_geo); loaded automatically from the run's rxft_meta.json.

Switches (environment variables; defaults in RUN SETTINGS below)
  EVAL_WEIGHTS=...                       checkpoint root (default: /media/PorableSSD/Work/Academic/PHDWork/development/MTL/WoodScapeGit/omnidet/pretrained_models/MyModel/MTL_bitrates/)
  EVAL_SPLITS=val,test                   which SynWoodScape lists to evaluate
  EVAL_BN_RECALIB_N=250                  training TUPLES for BatchNorm statistics (0 = off)
  EVAL_BN_RECALIB_SCOPE=no_stats|all     'all' also re-estimates the baseline (strict like-for-like)
  EVAL_PERCEPTION_INPUT=float|8bit       what the receiver-side encoder pass sees
  EVAL_SKIP_CODECS=1                     skip HEVC / JPEG2000
  EVAL_SKIP_PROFILE=1                    skip parameter / GMAC / latency profiling
  EVAL_VAL_FILE / EVAL_TEST_FILE         evaluation lists (default: val_file of the config, data/test.txt)
  EVAL_BN_FILE                           frames used for BatchNorm statistics (default: train_file)
"""
import os
import sys
import math
import time
import json
import yaml
import shutil
import socket
import pickle
import copy
import platform
import datetime
import subprocess
import tempfile
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from data_loader.synwoodscape_loader import SynWoodScapeRawDataset
from train_utils.detection_utils import non_max_suppression, xywh2xyxy, get_batch_statistics, ap_per_class
from main import collect_args
from models.detection_decoderORG import YoloDecoder
from models.resnet import ResnetEncoder
from models.semantic_decoder import SemanticDecoder
from models.compressAI.ldmic import Multi_LDMIC, Multi_JointContextTransfer
from utils import Tupperware

try:
    from thop import profile as thop_profile
except ImportError:  # profiling is optional
    thop_profile = None


def _flag(name, default=False):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes")


# ===== RUN SETTINGS: edit here (an environment variable of the same name overrides) =====
RUN_SPLITS = "val,test"
RUN_BN_RECALIB_N = 250            # tuples = 1,000 images, as the 1,000 images of Part B
RUN_BN_RECALIB_SCOPE = "no_stats"
RUN_BN_RECALIB_SEED = 0
RUN_PERCEPTION_INPUT = "float"    # receiver side: x_hat as the decoder outputs it ("8bit": stored image)
# =========================================================================================

SPLITS = [s.strip() for s in os.environ.get("EVAL_SPLITS", RUN_SPLITS).split(",") if s.strip()]
BN_RECALIB_N = int(os.environ.get("EVAL_BN_RECALIB_N", RUN_BN_RECALIB_N))
BN_RECALIB_SCOPE = os.environ.get("EVAL_BN_RECALIB_SCOPE", RUN_BN_RECALIB_SCOPE)
BN_RECALIB_SEED = int(os.environ.get("EVAL_BN_RECALIB_SEED", RUN_BN_RECALIB_SEED))
PERCEPTION_INPUT = os.environ.get("EVAL_PERCEPTION_INPUT", RUN_PERCEPTION_INPUT)
# Receiver-side BatchNorm [C13]: "trained" = the receiver encoder uses the statistics saved in the
# checkpoint (estimated on original images); "xhat" = the receiver gets its own copy of the encoder
# whose BatchNorm statistics are re-estimated from reconstructions x_hat of training tuples (no
# weight changes; the sender encoder that produces the bitstream is untouched). "xhat" corresponds
# to the Part B protocol, where the statistics covered the second encoder pass on x_hat.
RX_BN = os.environ.get("EVAL_RX_BN", "trained")
assert RX_BN in ("trained", "xhat"), RX_BN
# Fine-tuned receiver [C14]: EVAL_RX_FT=1 loads <run>_RXFT/models/weights_<n>/ (written by
# finetune_svs_receiver.py) for the receiver encoder, fusion and heads; the sender is unchanged.
RX_FT = _flag("EVAL_RX_FT")
# [C15] EVAL_CODEC_FT=1: each HEVC row also gets an "(adapted)" row, with the baseline fine-tuned on
# training frames decoded at that QP (finetune_svs_receiver.py, FT_MODE=hevc).
CODEC_FT = _flag("EVAL_CODEC_FT")
# [C16] EVAL_RX_FT_SUFFIX selects which fine-tuned receiver to load ("_RXFT": heavy fusion as trained,
# "_RXFT_SLIM": slim fusion). EVAL_SX_SLIM=1: the sender-side perception uses the slim fusion and
# heads retrained with the sender encoder frozen (<run>_SXSLIM); bitstream and rates unchanged.
RX_FT_SUFFIX = os.environ.get("EVAL_RX_FT_SUFFIX", "_RXFT")
SX_SLIM = _flag("EVAL_SX_SLIM")
# [C18] Which sender / end-to-end run to load: EVAL_SX_SUFFIX (_SXSLIM, _SXGEO, _SXNOFUSION; setting it
# implies EVAL_SX_SLIM=1) and EVAL_E2E_SUFFIX (_E2E, _E2E_GEO).
SX_SUFFIX = os.environ.get("EVAL_SX_SUFFIX", "_SXSLIM")
SX_SLIM = SX_SLIM or "EVAL_SX_SUFFIX" in os.environ
E2E_SUFFIX = os.environ.get("EVAL_E2E_SUFFIX", "_E2E")
# [C17] EVAL_E2E=1: projection, codec and receiver from <run>_E2E (finetune_svs_receiver.py, FT_MODE=e2e).
# The sender encoder is the checkpoint's, so __S is unchanged; the bitstream, rates, PSNR and
# __R change. Overrides EVAL_RX_FT for the receiver.
E2E = _flag("EVAL_E2E")
SKIP_CODECS = _flag("EVAL_SKIP_CODECS")
SKIP_PROFILE = _flag("EVAL_SKIP_PROFILE")
BN_FILE = os.environ.get("EVAL_BN_FILE", "")   # frames for BatchNorm statistics (default: train_file)
VOID_INDEX = int(os.environ.get("EVAL_VOID_INDEX", "0"))
assert BN_RECALIB_SCOPE in ("no_stats", "all"), BN_RECALIB_SCOPE
assert PERCEPTION_INPUT in ("float", "8bit"), PERCEPTION_INPUT
N_CAM = 4
DEFAULT_WEIGHTS = "/media/PorableSSD/Work/Academic/PHDWork/development/MTL/WoodScapeGit/omnidet/pretrained_models/MyModel/MTL_bitrates/"

# --------------------------------------------------------------------------------------
# Models (folders and checkpoints copied from evaluate_mic_mtl.py)
# --------------------------------------------------------------------------------------
BASELINE = "SVS_Baseline_no_compression"
DL_MODEL_CONFIGS = [
    dict(name=BASELINE, kind="baseline", lam=None,
         pretrained_path_segment="res50_no_mic_no_compression_FR_2Classes_yolo3_SVS", batch_suffix="99/39700"),
    dict(name="SVS_JPC_lam0.08", kind="svs", lam=0.08,
         pretrained_path_segment="res50_LDMIC0.08_FR_2Classes_yolo3_SVS", batch_suffix="117/46900"),
    dict(name="SVS_JPC_lam0.05", kind="svs", lam=0.05,
         pretrained_path_segment="res50_LDMIC0.05_FR_2Classes_yolo3_SVS", batch_suffix="115/46200"),
    dict(name="SVS_JPC_lam0.03", kind="svs", lam=0.03,
         pretrained_path_segment="res50_LDMIC0.03_FR_2Classes_yolo3_SVS", batch_suffix="111/44600"),
    dict(name="SVS_JPC_lam0.01", kind="svs", lam=0.01,
         pretrained_path_segment="res50_LDMIC0.01_FR_2Classes_yolo3_SVS", batch_suffix="107/42700"),
]
HEVC_QPS = {"HM HEVC QP31": 31, "HM HEVC QP33": 33, "HM HEVC QP36": 36, "HM HEVC QP39": 39,
            "HM HEVC QP42": 42, "HM HEVC QP45": 45}
if os.environ.get("EVAL_HEVC_QPS"):                       # e.g. 24,27,30,33,36,39,42,45 for 0.1-1 bpp
    HEVC_QPS = {f"HM HEVC QP{q}": int(q) for q in os.environ["EVAL_HEVC_QPS"].split(",") if q.strip()}
# [C22] Sender-side evaluation of jointly trained runs (finetune_svs_receiver.py FT_MODE=joint):
# EVAL_JOINT_RUNS=tag1,tag2 (folders under <MTL_bitrates>/JOINT/), EVAL_SENDER_ONLY=1 (no receiver pass),
# EVAL_CODEC_PERC_JOINT=BASE_FT: codec rows also perceived by that jointly trained perception-only run.
JOINT_RUNS = [t.strip() for t in os.environ.get("EVAL_JOINT_RUNS", "").split(",") if t.strip()]
SENDER_ONLY = _flag("EVAL_SENDER_ONLY")
CODEC_PERC_JOINT = os.environ.get("EVAL_CODEC_PERC_JOINT", "")
NO_RELEASED = _flag("EVAL_NO_RELEASED")                    # skip the four released SVS models
J2K_QS = {"JPEG2000 cl10": 10, "JPEG2000 cl15": 15, "JPEG2000 cl25": 25, "JPEG2000 cl50": 50,
          "JPEG2000 cl75": 75, "JPEG2000 cl95": 95}
if os.environ.get("EVAL_CODECS", "all") == "hevc":
    J2K_QS = {}


# --------------------------------------------------------------------------------------
# Pixels and metrics
# --------------------------------------------------------------------------------------
def to_8bit(x):
    return (x.clamp(0, 1) * 255).round() / 255


def tensor_to_pil(x):
    x = x.detach().cpu()
    if x.dim() == 4:
        x = x.squeeze(0)
    return Image.fromarray((x.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy())


def psnr(a, b):
    mse = F.mse_loss(a.float(), b.float()).item()
    return float("inf") if mse == 0 else -10 * math.log10(mse)


def seg_confusion(pred, gt, k):
    p = pred.reshape(-1).cpu().numpy().astype(np.int64)
    g = gt.reshape(-1).cpu().numpy().astype(np.int64)
    m = (g >= 0) & (g < k)
    return np.bincount(k * g[m] + p[m], minlength=k * k).reshape(k, k)


def summarize(det_records, seg_confs):
    res = {"mAP": float("nan"), "AP_per_class": {}, "mIoU_all": float("nan"), "mIoU_no_void": float("nan")}
    recs = [r for r in det_records.values() if r is not None]
    stats = [s for r in recs for s in r["stats"]]
    labels = [c for r in recs for c in r["target_cls"]]
    if stats and labels:
        tp, conf, pcls = [np.concatenate(x, 0) for x in zip(*stats)]
        _, _, ap, _, ap_cls = ap_per_class(tp, conf, pcls, np.array(labels))
        res["mAP"] = float(np.mean(ap))
        res["AP_per_class"] = {int(c): float(a) for c, a in zip(ap_cls, ap)}
    res["n_tuples_det"] = len(recs)
    if seg_confs:
        conf = sum(seg_confs.values())
        tp = np.diag(conf).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tp / (conf.sum(0) + conf.sum(1) - tp)
        res["mIoU_all"] = float(np.nanmean(iou))
        res["mIoU_no_void"] = float(np.nanmean(np.delete(iou, VOID_INDEX)))
        res["IoU_per_class"] = [float(x) for x in iou]
    return res


# --------------------------------------------------------------------------------------
# Loading [C2, C3]
# --------------------------------------------------------------------------------------
ENTROPY_BUFFERS = ("_quantized_cdf", "_offset", "_cdf_length", "scale_table")


def _resize_entropy_buffers(module, sd):
    """CompressAI entropy buffers change size after update(); match the checkpoint's sizes."""
    own = module.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape != v.shape and k.rsplit(".", 1)[-1] in ENTROPY_BUFFERS:
            parent = module.get_submodule(k.rsplit(".", 1)[0]) if "." in k else module
            parent.register_buffer(k.rsplit(".", 1)[-1], torch.zeros_like(v))


def bn_per_image(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None


def load_strict(module, path, device, filter_extra=False):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    sd = torch.load(path, map_location=device)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    if filter_extra:  # encoder.pth also stores height / width
        own = module.state_dict()
        sd = {k: v for k, v in sd.items() if k in own}
    _resize_entropy_buffers(module, sd)
    # nn.Module's loader is called directly: the CompressAI modules override load_state_dict()
    # without a 'strict' argument; their buffer resizing is done above.
    missing, unexpected = nn.Module.load_state_dict(module, sd, strict=False)
    stat_keys = [k for k in missing if k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    other = [k for k in missing if k not in stat_keys]
    if other or unexpected:
        raise RuntimeError(f"{path}\n  missing: {other}\n  unexpected: {unexpected}")
    no_stats = sorted({k.rsplit(".", 1)[0] for k in stat_keys})
    if no_stats:
        bn_all = [n for n, m in module.named_modules() if isinstance(m, nn.BatchNorm2d)]
        if set(no_stats) != set(bn_all):
            raise RuntimeError(f"{path}: running statistics missing for only some BatchNorm layers")
        bn_per_image(module)
        return "per_image (checkpoint has no running statistics)"
    return "running_stats"


def run_dir_of(cfg, args):
    return os.path.join(args.pretrained_weights, cfg["pretrained_path_segment"], "models",
                        f"weights_{cfg['batch_suffix']}")


def joint_dir(args, tag):
    return latest_weights_dir(os.path.join(args.pretrained_weights, "JOINT", tag, "models"))


def load_joint_bundle(cfg, args, device):
    """A jointly trained sender (FT_MODE=joint) [C22]."""
    d = joint_dir(args, cfg["tag"])
    if d is None or not os.path.exists(os.path.join(d, "joint_meta.json")):
        return None
    meta = json.load(open(os.path.join(d, "joint_meta.json")))
    enc = ResnetEncoder(num_layers=50, pretrained=False).to(device).eval()
    bn = {"encoder": load_strict(enc, os.path.join(d, "encoder.pth"), device, filter_extra=True)}
    det = YoloDecoder(enc.num_ch_enc, args).to(device).eval()
    load_strict(det, os.path.join(d, "detection.pth"), device)
    sem = SemanticDecoder(enc.num_ch_enc, n_classes=args.semantic_num_classes).to(device).eval()
    load_strict(sem, os.path.join(d, "semantic.pth"), device)
    fusion = make_fusion(meta["fusion"], enc.num_ch_enc[-1], device)
    if fusion is not None:
        load_strict(fusion, os.path.join(d, "feature_fusion.pth"), device)
        fusion.eval()
    proj = comp = None
    if not meta.get("nocodec"):
        proj = nn.Conv2d(enc.num_ch_enc[-2], 192, kernel_size=1).to(device).eval()
        load_strict(proj, os.path.join(d, "compression_projection.pth"), device)
        comp = Multi_LDMIC(N=192, M=192, decode_atten=Multi_JointContextTransfer).to(device).eval()
        attach_yhat_tap(comp)
        if meta.get("codec_geo"):
            add_codec_geo(comp, CAMS)
        load_strict(comp, os.path.join(d, "decoder.pth"), device)
        comp.eval()
    side = None
    if meta.get("side"):
        side = SideAnalysis().to(device)
        load_strict(side, os.path.join(d, "side_analysis.pth"), device)
        side.eval()
    size_mb = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".pth")) / 2 ** 20
    cfg["lam"] = meta.get("lambda_")
    return dict(run_dir=d, enc=enc, det=det, sem=sem, fusion=fusion, proj=proj, comp=comp, side=side, bn_mode=bn,
                size_mb=size_mb, joint_meta={k: meta.get(k) for k in ("tag", "source_name", "lambda_", "fusion",
                                                                       "codec_geo", "nocodec", "epochs", "side",
                                                                       "detach_codec", "init")})


def load_bundle(cfg, args, device):
    if cfg["kind"] == "joint":
        return load_joint_bundle(cfg, args, device)
    d = run_dir_of(cfg, args)
    if not os.path.isdir(d):
        return None
    enc = ResnetEncoder(num_layers=50, pretrained=False).to(device).eval()
    bn = {"encoder": load_strict(enc, os.path.join(d, "encoder.pth"), device, filter_extra=True)}
    det = YoloDecoder(enc.num_ch_enc, args).to(device).eval()
    bn["det"] = load_strict(det, os.path.join(d, "detection.pth"), device)
    sem = SemanticDecoder(enc.num_ch_enc, n_classes=args.semantic_num_classes).to(device).eval()
    bn["sem"] = load_strict(sem, os.path.join(d, "semantic.pth"), device)
    fusion = proj = comp = None
    if cfg["kind"] == "svs":
        fusion = Multi_JointContextTransfer(enc.num_ch_enc[-1]).to(device).eval()
        bn["fusion"] = load_strict(fusion, os.path.join(d, "feature_fusion.pth"), device)
        proj = nn.Conv2d(enc.num_ch_enc[-2], 192, kernel_size=1).to(device).eval()
        load_strict(proj, os.path.join(d, "compression_projection.pth"), device)
        comp = Multi_LDMIC(N=192, M=192, decode_atten=Multi_JointContextTransfer).to(device).eval()
        attach_yhat_tap(comp)
        bn["comp"] = load_strict(comp, os.path.join(d, "decoder.pth"), device)
    files = ["encoder.pth", "detection.pth", "semantic.pth", "decoder.pth", "feature_fusion.pth",
             "compression_projection.pth"]
    size_mb = sum(os.path.getsize(os.path.join(d, f)) for f in files if os.path.exists(os.path.join(d, f))) / 2 ** 20
    return dict(run_dir=d, enc=enc, det=det, sem=sem, fusion=fusion, proj=proj, comp=comp, bn_mode=bn,
                size_mb=size_mb)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def check_lists(args, named_lists):
    """Stop before any model runs if a listed frame is missing for any of the four cameras."""
    cams = ("FV", "MVL", "MVR", "RV")
    for name, lf in named_lists.items():
        frames = sorted({l.strip().split(".")[0].split("_")[0] for l in open(lf) if l.strip()})
        missing = [f for f in frames for c in cams
                   if not os.path.exists(os.path.join(args.dataset_dir, args.rgb_images, f"{f}_{c}.png"))]
        if missing:
            sys.exit(f"[{name}] {lf}: {len(missing)} of {4 * len(frames)} camera images are missing in "
                     f"{os.path.join(args.dataset_dir, args.rgb_images)} (first: {missing[:5]}).\n"
                     f"Use lists that match the local data (see RUN_SVS_EVAL.md).")
        print(f"[lists] {name}: {lf} ({len(frames)} frames, all four cameras present)")


def make_dataset(args, list_file):
    ds = SynWoodScapeRawDataset(data_path=args.dataset_dir, path_file=list_file, is_train=False, config=args)
    return ds


def make_loader(ds, args, indices=None):
    sub = ds if indices is None else torch.utils.data.Subset(ds, indices)
    return DataLoader(sub, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True,
                      drop_last=False, collate_fn=ds.collate_fn)


def tuple_images(inputs, device):
    """Four camera images as one batch, camera-major (FV, MVL, MVR, RV), 8-bit."""
    return to_8bit(torch.cat([inputs[("color", 0, 0, c)] for c in CAMS], 0).to(device))


def tuple_targets(inputs, img_size):
    parts = []
    for i, c in enumerate(CAMS):
        t = inputs.get(("detection_labels", 0, c))
        if t is not None and t.numel():
            t = t.view(-1, 6).cpu().clone()
            t[:, 0] = i
            parts.append(t)
    if not parts:
        return torch.empty((0, 6))
    t = torch.cat(parts, 0)
    t[:, 2:6] = xywh2xyxy(t[:, 2:6])
    t[:, [2, 4]] *= img_size[0]
    t[:, [3, 5]] *= img_size[1]
    return t


def tuple_semantic(inputs, device):
    labs = [inputs.get(("semantic_labels", 0, 0, c)) for c in CAMS]
    if any(l is None for l in labs):
        return None
    lab = torch.cat(labs, 0).to(device)
    return lab.squeeze(1) if lab.dim() == 4 else lab


# --------------------------------------------------------------------------------------
# Forward passes
# --------------------------------------------------------------------------------------
@torch.no_grad()
def fuse(bundle, f):
    f = list(f)
    if bundle["fusion"] is not None:
        f[-1] = bundle["fusion"](f[-1], N_CAM)
    return f


@torch.no_grad()
def features(bundle, x4):
    return fuse(bundle, bundle["enc"](x4))


@torch.no_grad()
def perceive(bundle, feats, targets, seg_gt, img_size, args):
    yolo = bundle["det"](feats, img_dim=img_size)["yolo_outputs"]
    nms = non_max_suppression(yolo, conf_thres=args.detection_conf_thres, nms_thres=args.detection_nms_thres)
    det = {"stats": get_batch_statistics(nms, targets.clone(), iou_threshold=0.5, args=args),
           "target_cls": targets[:, 1].tolist() if targets.numel() else []}
    conf = None
    if seg_gt is not None:
        pred = bundle["sem"](feats)[("semantic", 0)].argmax(1)
        conf = seg_confusion(pred, seg_gt, args.semantic_num_classes)
    return det, conf


@torch.no_grad()
def compress(bundle, x4, h, w):
    f = bundle["enc"](x4)
    y = project(bundle, f[-2], x4)
    out = bundle["comp"](y, num_camera=N_CAM)
    lk = out["likelihoods"]
    per_cam = sum((-torch.log2(l)).flatten(1).sum(1) for l in lk.values()) / (h * w)  # bits / pixel, per camera
    x_hat = out["x_hat"]
    xh8 = to_8bit(x_hat)
    row = {"BPP": float(per_cam.mean()), "BPP_per_camera": {c: float(b) for c, b in zip(CAMS, per_cam)},
           "PSNR": float(np.mean([psnr(x4[i:i + 1], xh8[i:i + 1]) for i in range(N_CAM)])),
           "MS_SSIM": float(ms_ssim(x4, xh8, data_range=1.0, size_average=True).item())}
    return f, x_hat, row


class SlimFusion(nn.Module):
    """Cross-view fusion on a 256-channel projection of F4 [C16]:
    x + W_up * J(W_down * x), with the LDMIC joint context transfer J at 256 channels.
    W_up starts at zero, so the module starts as the identity (no fusion) and learns what it adds.
    About 6 M parameters instead of 351 M."""

    def __init__(self, c_in=2048, c=256):
        super().__init__()
        self.down = nn.Conv2d(c_in, c, 1)
        self.jct = Multi_JointContextTransfer(c)
        self.up = nn.Conv2d(c, c_in, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, num_camera):
        return x + self.up(self.jct(self.down(x), num_camera))




# [C18] Geometry-aware fusion: each camera attends only to its two overlapping neighbours, with a
# positional encoding of the viewing direction of every feature position in the vehicle frame.
CAM_YAW_DEG = {"FV": 0.0, "MVL": 90.0, "MVR": -90.0, "RV": 180.0}          # nominal mounting yaw
CAM_NEIGHBOURS = {"FV": ("MVL", "MVR"), "MVL": ("FV", "RV"), "MVR": ("FV", "RV"), "RV": ("MVL", "MVR")}
GEO_FOV_DEG = float(os.environ.get("GEO_FOV_DEG", "190"))   # equidistant model: image width spans this FoV
GEO_SENSOR_WH = (1280, 966)                                 # native SynWoodScape resolution


def fisheye_rays(h, w, device):
    """Unit viewing direction (camera frame: x right, y down, z forward) of each cell of an h x w
    feature map, for an equidistant fisheye with the principal point at the image centre. Nominal
    model only: no calibration file is used."""
    sw, sh = GEO_SENSOR_WH
    f = (sw / 2) / math.radians(GEO_FOV_DEG / 2)
    ys = ((torch.arange(h, device=device) + 0.5) / h * sh - sh / 2)
    xs = ((torch.arange(w, device=device) + 0.5) / w * sw - sw / 2)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    theta = (torch.sqrt(xx ** 2 + yy ** 2) / f).clamp(max=math.radians(GEO_FOV_DEG / 2 + 10))
    phi = torch.atan2(yy, xx)
    return torch.stack([torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)], 0)


class GeoFusion(nn.Module):
    """x_i + W_up * Attn(q = camera i, k/v = its two overlapping neighbours), at 256 channels.
    Queries and keys carry a Fourier encoding of the viewing direction in the vehicle frame
    (nominal yaw per camera plus a learned offset), so a corner position of one camera can find the
    positions of its neighbour that look the same way. W_up starts at zero (identity at start)."""

    def __init__(self, c_in=2048, c=256, heads=4, n_freq=4, cams=None, pool=1):
        super().__init__()
        self.cams = list(cams or ["FV", "MVL", "MVR", "RV"])
        self.pool = pool                       # attend on a pooled grid, upsample the update (large maps)
        self.down = nn.Conv2d(c_in, c, 1)
        self.norm_q = nn.LayerNorm(c)
        self.norm_kv = nn.LayerNorm(c)
        self.n_freq = n_freq
        self.pe = nn.Sequential(nn.Linear(3 + 6 * n_freq, c), nn.GELU(), nn.Linear(c, c))
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.yaw_offset = nn.Parameter(torch.zeros(len(self.cams)))
        self.up = nn.Conv2d(c, c_in, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.nbr = [[self.cams.index(n) for n in CAM_NEIGHBOURS[c_]] for c_ in self.cams]

    def encode_dirs(self, h, w, device):
        d = fisheye_rays(h, w, device).flatten(1)                    # (3, h*w), camera frame
        out = []
        for i, cam in enumerate(self.cams):
            yaw = math.radians(CAM_YAW_DEG[cam]) + self.yaw_offset[i]
            fwd = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)])
            right = torch.stack([torch.sin(yaw), -torch.cos(yaw), torch.zeros_like(yaw)])
            down = torch.tensor([0.0, 0.0, -1.0], device=device)
            v = right[:, None] * d[0] + down[:, None] * d[1] + fwd[:, None] * d[2]   # vehicle frame (x fwd, y left, z up)
            f = [v] + [g(math.pi * 2 ** k * v) for k in range(self.n_freq) for g in (torch.sin, torch.cos)]
            out.append(torch.cat(f, 0).t())                           # (h*w, 3 + 6 n_freq)
        return self.pe(torch.stack(out, 0))                           # (V, h*w, c)

    def forward(self, x, num_camera):
        V = num_camera
        assert V == len(self.cams), (V, self.cams)
        n, _, H, W = x.shape
        B = n // V                                                    # images are camera-major per batch
        z = self.down(x)                                              # (V*B, c, H, W)
        if self.pool > 1:
            z = F.avg_pool2d(z, self.pool, ceil_mode=True)
        h, w = z.shape[-2:]
        c = z.shape[1]
        t = z.flatten(2).transpose(1, 2).reshape(V, B, h * w, c)      # (V, B, hw, c)
        pe = self.encode_dirs(h, w, x.device)[:, None]                # (V, 1, hw, c)
        q = self.norm_q(t) + pe
        kv = self.norm_kv(t)
        out = []
        for i in range(V):
            j = self.nbr[i]
            k = torch.cat([kv[a] + pe[a] for a in j], 1)              # (B, 2hw, c)
            v = torch.cat([kv[a] for a in j], 1)
            o, _ = self.attn(q[i], k, v, need_weights=False)          # (B, hw, c)
            out.append(o)
        o = torch.stack(out, 0).reshape(V * B, h * w, c).transpose(1, 2).reshape(V * B, c, h, w)
        if (h, w) != (H, W):
            o = F.interpolate(o, size=(H, W), mode="bilinear", align_corners=False)
        return x + self.up(o)


def add_codec_geo(comp, cams=None):
    """[C19] Geometry-aware attention inside the LDMIC joint decoder: after each of its two joint
    context transfers (on y_hat, 18 x 34, and on the mid-features, 72 x 136, attended at 18 x 34),
    a GeoFusion block lets every camera read its two overlapping neighbours by viewing direction.
    Added as comp.geo_3 / comp.geo_4 with forward hooks, so the released keys stay unchanged; the
    blocks start as the identity, so the codec starts exactly as trained."""
    c3 = comp.decoder_1[0].in_channels if hasattr(comp.decoder_1[0], "in_channels") else 192
    c4 = comp.decoder_2[0].in_channels if hasattr(comp.decoder_2[0], "in_channels") else 192
    dev = next(comp.parameters()).device
    comp.geo_3 = GeoFusion(c3, c=128, cams=cams).to(dev)
    comp.geo_4 = GeoFusion(c4, c=128, cams=cams, pool=4).to(dev)
    comp.atten_3.register_forward_hook(lambda mod, inp, out: comp.geo_3(out, inp[1]))
    comp.atten_4.register_forward_hook(lambda mod, inp, out: comp.geo_4(out, inp[1]))
    return comp


# [C20] Latent skip: the receiver adds the decoded latent y_hat (a projection of the SENDER's F3
# features, 192 x 18 x 34) to its own F3 of x_hat through a small adapter, then computes F4 from it.
# The receiver thus perceives from the pixels AND from the transmitted features, at no extra bits.
_YHAT = None                                     # y_hat of the latest codec forward (set by a hook)


def attach_yhat_tap(comp):
    """Record the decoder input y_hat (rounded latent, straight-through gradient) on every forward."""
    def tap(mod, inp):
        global _YHAT
        _YHAT = inp[0]
    comp.atten_3.register_forward_pre_hook(tap)
    return comp


class LatentSkip(nn.Module):
    """y_hat (M channels) -> F3 (1024 channels): 1x1, ReLU, 1x1; the last layer starts at zero."""

    def __init__(self, m=192, c_out=1024, c=256):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(m, c, 1), nn.ReLU(inplace=True), nn.Conv2d(c, c_out, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, y_hat, size):
        s = self.net(y_hat)
        return s if s.shape[-2:] == size else F.interpolate(s, size=size, mode="bilinear", align_corners=False)


def rx_encode(rx, x):
    """Receiver encoder features of x_hat; with a latent skip, F3 also gets the transmitted y_hat."""
    if rx.get("skip") is None:
        return rx["enc"](x)
    e = rx["enc"].encoder
    f0 = e.relu(e.bn1(e.conv1(x)))
    f1 = e.layer1(e.maxpool(f0))
    f2 = e.layer2(f1)
    f3 = e.layer3(f2)
    f3 = f3 + rx["skip"](_YHAT, f3.shape[-2:])
    return [f0, f1, f2, f3, e.layer4(f3)]


class SideAnalysis(nn.Module):
    """[C23] Light image analysis branch for the codec: four 5x5 stride-2 convolutions with GDN
    (3 -> 64 -> 96 -> 128 -> 192 channels, about 2.8 GMACs per 544x288 image). Its output is ADDED to
    the F3 projection, so the latent carries pixel detail the perception features do not keep. The
    last layer starts at zero, so a run starts exactly where its source model is."""

    def __init__(self, m=192):
        super().__init__()
        from models.compressAI.layers import GDN
        from models.AEs.utils import conv
        self.net = nn.Sequential(conv(3, 64, kernel_size=5, stride=2), GDN(64),
                                 conv(64, 96, kernel_size=5, stride=2), GDN(96),
                                 conv(96, 128, kernel_size=5, stride=2), GDN(128),
                                 conv(128, m, kernel_size=5, stride=2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, size):
        s = self.net(x)
        return s if s.shape[-2:] == size else F.interpolate(s, size=size, mode="bilinear", align_corners=False)


def project(bundle, f3, x4):
    """Codec latent y: projection of the shared F3, plus the side analysis branch if present [C23]."""
    y = bundle["proj"](f3)
    if bundle.get("side") is not None:
        y = y + bundle["side"](x4, y.shape[-2:])
    return y


class AdapterFusion(nn.Module):
    """[C24] Control for the slim fusion: the same 2048 -> 256 -> 2048 bottleneck with two residual
    3x3 blocks per camera, but NO exchange between cameras (about 3.4 M parameters)."""

    def __init__(self, c_in=2048, c=256):
        super().__init__()
        self.down = nn.Conv2d(c_in, c, 1)
        self.body = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
                                                 nn.Conv2d(c, c, 3, padding=1)) for _ in range(2)])
        self.up = nn.Conv2d(c, c_in, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, num_camera):
        z = self.down(x)
        for blk in self.body:
            z = z + blk(z)
        return x + self.up(z)


def make_fusion(kind, c_in, device):
    """'heavy' (released module), 'slim' [C16], 'geo' [C18] or 'none' (no fusion)."""
    if kind == "none":
        return None
    if kind == "geo":
        return GeoFusion(c_in, cams=CAMS).to(device)
    if kind == "adapter":
        return AdapterFusion(c_in).to(device)
    return (SlimFusion(c_in) if kind == "slim" else Multi_JointContextTransfer(c_in)).to(device)


def load_ft_dir(d, b, args, device, enc=None):
    """Encoder (unless given), fusion and heads saved by finetune_svs_receiver.py in folder d."""
    meta_p = os.path.join(d, "rxft_meta.json")
    kind = json.load(open(meta_p)).get("fusion", "heavy") if os.path.exists(meta_p) else "heavy"
    if enc is None:
        enc = copy.deepcopy(b["enc"])
        load_strict(enc, os.path.join(d, "encoder.pth"), device, filter_extra=True)
    fusion = make_fusion(kind, b["enc"].num_ch_enc[-1], device)
    if fusion is not None:
        load_strict(fusion, os.path.join(d, "feature_fusion.pth"), device)
    det = YoloDecoder(b["enc"].num_ch_enc, args).to(device)
    load_strict(det, os.path.join(d, "detection.pth"), device)
    sem = SemanticDecoder(b["enc"].num_ch_enc, n_classes=args.semantic_num_classes).to(device)
    load_strict(sem, os.path.join(d, "semantic.pth"), device)
    out = dict(enc=enc, fusion=fusion, det=det, sem=sem)
    if os.path.exists(meta_p) and json.load(open(meta_p)).get("latent_skip"):
        out["skip"] = LatentSkip(c_out=b["enc"].num_ch_enc[-2]).to(device)
        load_strict(out["skip"], os.path.join(d, "latent_skip.pth"), device)
    for m in out.values():
        if m is not None:
            m.eval()
    return out, kind


def latest_weights_dir(root):
    subs = sorted(os.listdir(root), key=lambda n: int(n.split("_")[-1])) if os.path.isdir(root) else []
    return os.path.join(root, subs[-1]) if subs else None


@torch.no_grad()
def clone_heads(b, args, device):
    """New fusion / detection / segmentation modules holding copies of b's weights (the YOLO head
    keeps a reference to the config object, so it cannot be deep-copied)."""
    fusion = Multi_JointContextTransfer(b["enc"].num_ch_enc[-1]).to(device)
    fusion.load_state_dict(b["fusion"].state_dict())
    det = YoloDecoder(b["enc"].num_ch_enc, args).to(device)
    det.load_state_dict(b["det"].state_dict())
    sem = SemanticDecoder(b["enc"].num_ch_enc, n_classes=args.semantic_num_classes).to(device)
    sem.load_state_dict(b["sem"].state_dict())
    return dict(fusion=fusion, det=det, sem=sem)


def load_codec_ft(qp, args, device, base):
    """Baseline encoder and heads fine-tuned on HEVC-decoded frames at this QP [C15]."""
    root = os.path.join(args.pretrained_weights,
                        f"{DL_MODEL_CONFIGS[0]['pretrained_path_segment']}_RXFT_HEVC_QP{qp}", "models")
    subs = sorted(os.listdir(root), key=lambda n: int(n.split("_")[-1])) if os.path.isdir(root) else []
    if not subs:
        return None, f"no adapted baseline in {root}"
    d = os.path.join(root, subs[-1])
    enc = ResnetEncoder(num_layers=50, pretrained=False).to(device)
    load_strict(enc, os.path.join(d, "encoder.pth"), device, filter_extra=True)
    det = YoloDecoder(enc.num_ch_enc, args).to(device)
    load_strict(det, os.path.join(d, "detection.pth"), device)
    sem = SemanticDecoder(enc.num_ch_enc, n_classes=args.semantic_num_classes).to(device)
    load_strict(sem, os.path.join(d, "semantic.pth"), device)
    for m in (enc, det, sem):
        m.eval()
    return dict(enc=enc, fusion=None, det=det, sem=sem), f"baseline fine-tuned on HEVC QP{qp} frames ({d})"


def load_rx_ft(cfg, args, device, b):
    """Receiver modules fine-tuned by finetune_svs_receiver.py [C14, C16]."""
    root = os.path.join(args.pretrained_weights, cfg["pretrained_path_segment"] + RX_FT_SUFFIX, "models")
    d = latest_weights_dir(root)
    if d is None:
        return None, f"no fine-tuned receiver in {root}"
    rx, kind = load_ft_dir(d, b, args, device)
    return rx, f"fine-tuned receiver ({kind} fusion) from {d}"


def load_sx_slim(cfg, args, device, b):
    """Sender-side slim fusion and heads; the sender encoder stays the checkpoint's [C16]."""
    d = latest_weights_dir(os.path.join(args.pretrained_weights, cfg["pretrained_path_segment"] + SX_SUFFIX, "models"))
    if d is None:
        return None, f"no sender run {SX_SUFFIX}"
    tx, kind = load_ft_dir(d, b, args, device, enc=b["enc"])
    return tx, f"sender {kind} fusion and heads from {d}"


def load_e2e(cfg, args, device, b):
    """Task-aware codec and its receiver, trained end to end [C17]. Replaces b["proj"] and b["comp"]."""
    d = latest_weights_dir(os.path.join(args.pretrained_weights, cfg["pretrained_path_segment"] + E2E_SUFFIX, "models"))
    if d is None:
        return None, "no end-to-end run"
    load_strict(b["proj"], os.path.join(d, "compression_projection.pth"), device)
    meta_p = os.path.join(d, "rxft_meta.json")
    if os.path.exists(meta_p) and json.load(open(meta_p)).get("codec_geo"):
        add_codec_geo(b["comp"], CAMS)
    load_strict(b["comp"], os.path.join(d, "decoder.pth"), device)
    b["proj"].eval()
    b["comp"].eval()
    rx, kind = load_ft_dir(d, b, args, device)
    return rx, f"task-aware codec and receiver ({kind} fusion) from {d}"


def rx_bn_from_xhat(bundle, args, device, n_tuples, seed):
    """Receiver encoder copy with BatchNorm statistics from reconstructions only [C13]."""
    enc_rx = copy.deepcopy(bundle["enc"])
    for m in enc_rx.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            m.running_mean = torch.zeros(m.num_features, device=device)
            m.running_var = torch.ones(m.num_features, device=device)
            m.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=device)
            m.momentum = None
    ds = make_dataset(args, BN_FILE or args.train_file)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_tuples].tolist()
    h, w = int(args.input_height), int(args.input_width)
    bundle["enc"].eval()
    enc_rx.train()
    with torch.no_grad():
        for inputs in tqdm(make_loader(ds, args, idx), desc="receiver BN from x_hat"):
            x4 = tuple_images(inputs, device)
            _, x_hat, _ = compress(bundle, x4, h, w)          # sender encoder, trained statistics
            enc_rx(x_hat if PERCEPTION_INPUT == "float" else to_8bit(x_hat))
    enc_rx.eval()
    return enc_rx, f"receiver copy, statistics from x_hat of {len(idx)} training tuples (seed {seed})"


def bn_recalibrate(bundle, args, device, n_tuples, seed):
    enc = bundle["enc"]
    for m in enc.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            m.running_mean = torch.zeros(m.num_features, device=device)
            m.running_var = torch.ones(m.num_features, device=device)
            m.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=device)
            m.momentum = None
    ds = make_dataset(args, BN_FILE or args.train_file)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_tuples].tolist()
    h, w = int(args.input_height), int(args.input_width)
    enc.train()
    for inputs in tqdm(make_loader(ds, args, idx), desc="BN recalibration"):
        x4 = tuple_images(inputs, device)
        enc(x4)
        if bundle["comp"] is not None:  # the receiver-side pass sees x_hat
            enc.eval()
            _, x_hat, _ = compress(bundle, x4, h, w)
            enc.train()
            enc(x_hat if PERCEPTION_INPUT == "float" else to_8bit(x_hat))
    enc.eval()
    return f"running_stats (recalibrated on {len(idx)} training tuples = {4 * len(idx)} images, seed {seed})"


# --------------------------------------------------------------------------------------
# Codecs [C4]
# --------------------------------------------------------------------------------------
NAN_CODEC = {"BPP": np.nan, "PSNR": np.nan, "MS_SSIM": np.nan}


def _run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _quality(orig, png):
    rec = transforms.ToTensor()(Image.open(png).convert("RGB")).unsqueeze(0)
    o = orig.float().cpu().unsqueeze(0) if orig.dim() == 3 else orig.float().cpu()
    return psnr(o, rec), ms_ssim(o, rec, data_range=1.0).item()


def compress_hevc(orig, qp, hm, out_png):
    cfg_main, cfg_seq, enc_bin, dec_bin = hm
    pil = tensor_to_pil(orig)
    w, h = pil.size
    with tempfile.TemporaryDirectory() as d:
        png, yuv, bit, dyuv, out = (os.path.join(d, n) for n in ("in.png", "in.yuv", "b.bin", "d.yuv", "out.png"))
        pil.save(png)
        try:
            _run(["ffmpeg", "-y", "-loglevel", "error", "-i", png, "-s", f"{w}x{h}", "-pix_fmt", "yuv420p", yuv])
            _run([enc_bin, "-c", cfg_main, "-c", cfg_seq, "-i", yuv, "-wdt", str(w), "-hgt", str(h), "-f", "1",
                  f"--QP={qp}", "--ConformanceWindowMode=1", "-o", os.devnull, "-b", bit])
            bpp = os.path.getsize(bit) * 8 / (w * h)
            _run([dec_bin, "-b", bit, "-o", dyuv])
            _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
                  "-i", dyuv, out])
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"HEVC QP{qp} failed: {e}")
            return dict(NAN_CODEC)
        p, s = _quality(orig, out)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        shutil.copy(out, out_png)
        return {"BPP": bpp, "PSNR": p, "MS_SSIM": s}


def compress_j2k(orig, level, out_png):
    pil = tensor_to_pil(orig)
    w, h = pil.size
    with tempfile.TemporaryDirectory() as d:
        png, j2k, out = (os.path.join(d, n) for n in ("in.png", "e.j2k", "out.png"))
        pil.save(png)
        try:
            _run(["ffmpeg", "-y", "-loglevel", "error", "-i", png, "-c:v", "libopenjpeg",
                  "-compression_level", str(level), j2k])
            bpp = os.path.getsize(j2k) * 8 / (w * h)
            _run(["ffmpeg", "-y", "-loglevel", "error", "-i", j2k, "-pix_fmt", "rgb24", out])
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"JPEG2000 level {level} failed: {e}")
            return dict(NAN_CODEC)
        p, s = _quality(orig, out)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        shutil.copy(out, out_png)
        return {"BPP": bpp, "PSNR": p, "MS_SSIM": s}


def _codec_worker_init():
    torch.set_num_threads(1)  # one thread per worker process


def codec_worker(task):
    key, imgs, codec_dir, hm, cams = task  # imgs: list of 4 [3,H,W] tensors (8-bit levels)
    out = {}
    for name, qp in HEVC_QPS.items():
        rows = [compress_hevc(im, qp, hm, os.path.join(codec_dir, name, f"{key}_{c}.png")) for c, im in zip(cams, imgs)]
        out[name] = {m: float(np.nanmean([r[m] for r in rows])) for m in ("BPP", "PSNR", "MS_SSIM")}
    for name, lv in J2K_QS.items():
        rows = [compress_j2k(im, lv, os.path.join(codec_dir, name, f"{key}_{c}.png")) for c, im in zip(cams, imgs)]
        out[name] = {m: float(np.nanmean([r[m] for r in rows])) for m in ("BPP", "PSNR", "MS_SSIM")}
    return key, out


# --------------------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------------------
def _timer(device, n_warm=20, n_run=200):
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    def timeit(fn):
        for _ in range(n_warm):
            fn()
        sync()
        ts = []
        for _ in range(n_run):
            t = time.perf_counter()
            fn()
            sync()
            ts.append((time.perf_counter() - t) * 1e3)
        return float(np.median(ts)), float(np.percentile(ts, 95))
    return timeit


@torch.no_grad()
def profile_bundle(b, h, w, device):
    r = {"size_MB_all_checkpoints": b["size_mb"]}
    for name in ("enc", "fusion", "proj", "comp", "det", "sem"):
        m = b[name]
        r[f"Params_M_{name}"] = sum(p.numel() for p in m.parameters()) / 1e6 if m is not None else 0.0
    r["Params_M_total"] = sum(v for k, v in r.items() if k.startswith("Params_M_"))
    x4 = torch.rand(N_CAM, 3, h, w, device=device)
    img_size = [w, h]
    if thop_profile is not None:
        f = b["enc"](x4)
        r["GMACs_encoder_4cams"] = thop_profile(b["enc"], inputs=(x4,), verbose=False)[0] / 1e9
        if b["fusion"] is not None:
            r["GMACs_fusion_F4"] = thop_profile(b["fusion"], inputs=(f[-1], N_CAM), verbose=False)[0] / 1e9
            y = b["proj"](f[-2])
            r["GMACs_codec_forward"] = thop_profile(b["comp"], inputs=(y, N_CAM), verbose=False)[0] / 1e9
    timeit = _timer(device)
    r["encoder_4cams_ms"] = timeit(lambda: b["enc"](x4))

    tx = b.get("tx") or dict(enc=b["enc"], fusion=b["fusion"], det=b["det"], sem=b["sem"])
    for path, bundle in (("sender", tx), ("receiver", b.get("rx"))):
        if bundle is not None and bundle.get("fusion") is not None:
            r[f"Params_M_fusion_{path}_path"] = sum(p.numel() for p in bundle["fusion"].parameters()) / 1e6

    def sender():
        feats = fuse(tx, tx["enc"](x4))
        tx["det"](feats, img_dim=img_size)
        tx["sem"](feats)

    r["sender_pipeline_ms"] = timeit(sender)
    if b["comp"] is not None:
        # [C21] One network on the sender: perception plus everything needed to write the bitstream
        # (projection, hyper-analysis, quantisation, context model and entropy parameters; arithmetic
        # coding itself is not timed). The synthesis transform runs only where images are wanted.
        class _BitstreamEncoder(nn.Module):
            def __init__(self, proj, comp, side):
                super().__init__()
                self.proj, self.comp, self.side = proj, comp, side

            def forward(self, f3, x):
                c = self.comp
                y = self.proj(f3)
                if self.side is not None:
                    y = y + self.side(x, y.shape[-2:])
                params, _ = c.hyperprior(y)
                y_hat = c.gaussian_conditional.quantize(y, "dequantize")
                ctx = c.context_prediction(y_hat)
                params = F.interpolate(params, size=ctx.size()[-2:])
                return c.entropy_parameters(torch.cat([params, ctx], 1))

        bits = _BitstreamEncoder(b["proj"], b["comp"], b.get("side")).eval()
        f3 = b["enc"](x4)[-2]
        if thop_profile is not None:
            try:
                r["GMACs_bitstream_encoder_4cams"] = thop_profile(bits, inputs=(f3, x4), verbose=False)[0] / 1e9
            except Exception as e:                    # thop is optional; timing below does not need it
                r["GMACs_bitstream_encoder_4cams"] = f"n/a ({type(e).__name__})"
        r["Params_M_bitstream_encoder"] = (sum(p.numel() for p in b["proj"].parameters())
                                           + (sum(p.numel() for p in b["side"].parameters()) if b.get("side") is not None else 0)
                                           + sum(p.numel() for n_, p in b["comp"].named_parameters()
                                                 if n_.split(".")[0] in ("hyperprior", "context_prediction",
                                                                         "entropy_parameters"))) / 1e6
        r["bitstream_encoder_ms"] = timeit(lambda: bits(f3, x4))

        def sender_with_bits():
            f = tx["enc"](x4)
            feats = fuse(tx, f)
            tx["det"](feats, img_dim=img_size)
            tx["sem"](feats)
            bits(f[-2], x4)

        r["sender_with_bitstream_ms"] = timeit(sender_with_bits)
    if b["comp"] is not None and b.get("rx") is not None:
        def receiver():
            _, xh, _ = compress(b, x4, h, w)
            feats = fuse(b["rx"], rx_encode(b["rx"], xh))
            b["rx"]["det"](feats, img_dim=img_size)
            b["rx"]["sem"](feats)
        f4 = b["enc"](x4)[-1]
        if tx.get("fusion") is not None:
            r["fusion_F4_sender_ms"] = timeit(lambda: tx["fusion"](f4, N_CAM))
        r["receiver_pipeline_ms"] = timeit(receiver)
    r["latency_note"] = "(median, P95) ms, one four-camera tuple, forward passes, no arithmetic coding"
    return r


# --------------------------------------------------------------------------------------
# One split
# --------------------------------------------------------------------------------------
def dump(per_img_dir, name, det, seg, comp):
    with open(os.path.join(per_img_dir, name + ".pkl"), "wb") as f:
        pickle.dump({"det": det, "seg": seg, "comp": comp}, f)


@torch.no_grad()
def evaluate_split(args, split_name, list_file, out_root, device, profiled):
    h, w = int(args.input_height), int(args.input_width)
    img_size = [w, h]
    ds = make_dataset(args, list_file)
    keys = list(ds.unique_frame_indices)
    out_dir = os.path.join(out_root, f"SynWoodScape_{split_name}")
    per_img_dir = os.path.join(out_dir, "per_image")
    codec_dir = os.path.join(out_dir, "codec_decoded")
    os.makedirs(per_img_dir, exist_ok=True)
    summary = {"n_tuples": len(ds), "list_file": list_file, "models": {}, "codecs": {}}

    # ---- learned models ----
    configs = [c for c in DL_MODEL_CONFIGS if not (NO_RELEASED and c["kind"] == "svs")]
    configs += [dict(name=f"JOINT_{t}", kind="joint", lam=None, tag=t) for t in JOINT_RUNS]
    for cfg in configs:
        b = load_bundle(cfg, args, device)
        if b is None:
            if cfg["name"] == BASELINE:
                raise SystemExit(f"Baseline checkpoint not found: {run_dir_of(cfg, args)}")
            where = run_dir_of(cfg, args) if cfg["kind"] != "joint" else os.path.join(args.pretrained_weights, "JOINT", cfg["tag"])
            print(f"[skip] {cfg['name']}: not found: {where}")
            continue
        if BN_RECALIB_N and (BN_RECALIB_SCOPE == "all" or b["bn_mode"]["encoder"].startswith("per_image")):
            b["bn_mode"]["encoder"] = bn_recalibrate(b, args, device, BN_RECALIB_N, BN_RECALIB_SEED)
        b["enc_rx"] = b["enc"]
        if RX_BN == "xhat" and b["comp"] is not None:
            b["enc_rx"], b["bn_mode"]["encoder_receiver"] = rx_bn_from_xhat(b, args, device, BN_RECALIB_N or 250,
                                                                             BN_RECALIB_SEED)
        b["rx"] = dict(enc=b["enc_rx"], fusion=b["fusion"], det=b["det"], sem=b["sem"])
        b["tx"] = dict(enc=b["enc"], fusion=b["fusion"], det=b["det"], sem=b["sem"])
        if SENDER_ONLY or cfg["kind"] == "joint":
            b["rx"] = None                               # [C22] no second pass
        if SX_SLIM and cfg["kind"] == "svs" and b["comp"] is not None:
            tx, msg = load_sx_slim(cfg, args, device, b)
            if tx is None:
                print(f"[skip] {cfg['name']}: {msg}")
                del b
                continue
            b["tx"], b["bn_mode"]["sender_slim"] = tx, msg
        if RX_FT and b["rx"] is not None and b["comp"] is not None:
            rx, msg = load_rx_ft(cfg, args, device, b)
            if rx is None:
                print(f"[skip] {cfg['name']}: {msg}")
                del b
                continue
            b["rx"], b["bn_mode"]["receiver_finetuned"] = rx, msg
        if E2E and b["rx"] is not None and b["comp"] is not None:
            rx, msg = load_e2e(cfg, args, device, b)
            if rx is None:
                print(f"[skip] {cfg['name']}: {msg}")
                del b
                continue
            b["rx"], b["bn_mode"]["e2e"] = rx, msg
        print(f"  {cfg['name']}: BatchNorm {b['bn_mode']}")
        recs = {"S": ({}, {}), "R": ({}, {})}
        comp_rows = {}
        for i, inputs in enumerate(tqdm(make_loader(ds, args), desc=f"{split_name} {cfg['name']}")):
            key = keys[i]
            x4 = tuple_images(inputs, device)
            targets = tuple_targets(inputs, img_size)
            seg_gt = tuple_semantic(inputs, device)
            if b["comp"] is None:
                d, c = perceive(b, features(b, x4), targets, seg_gt, img_size, args)
                recs["S"][0][key], recs["S"][1][key] = d, c
                continue
            f0, x_hat, row = compress(b, x4, h, w)
            comp_rows[key] = row
            d, c = perceive(b["tx"], fuse(b["tx"], f0), targets, seg_gt, img_size, args)  # sender side
            recs["S"][0][key], recs["S"][1][key] = d, c
            if b["rx"] is None:
                continue
            xin = x_hat if PERCEPTION_INPUT == "float" else to_8bit(x_hat)
            d, c = perceive(b["rx"], fuse(b["rx"], rx_encode(b["rx"], xin)), targets, seg_gt, img_size, args)  # receiver
            recs["R"][0][key], recs["R"][1][key] = d, c
        rate = {}
        if comp_rows:
            for m in ("BPP", "PSNR", "MS_SSIM"):
                rate[m] = float(np.nanmean([r[m] for r in comp_rows.values()]))
            rate["BPP_per_camera"] = {c: float(np.mean([r["BPP_per_camera"][c] for r in comp_rows.values()]))
                                      for c in CAMS}
        meta = {"lambda": cfg["lam"], "checkpoint": b["run_dir"], "bn_mode": b["bn_mode"]}
        if "joint_meta" in b:
            meta["joint"] = b["joint_meta"]
        if b["comp"] is None:
            res = summarize(*recs["S"])
            res.update(meta)
            summary["models"][cfg["name"]] = res
            dump(per_img_dir, cfg["name"], recs["S"][0], {k: v for k, v in recs["S"][1].items() if v is not None}, {})
        else:
            for path, label in (("S", "sender (encoder features of the original images)"),
                                ("R", "receiver (encoder features of x_hat)")):
                if path == "R" and b["rx"] is None:
                    continue
                res = summarize(*recs[path])
                res.update(rate)
                res.update(meta)
                res["perception_path"] = label
                summary["models"][f"{cfg['name']}__{path}"] = res
                dump(per_img_dir, f"{cfg['name']}__{path}", recs[path][0],
                     {k: v for k, v in recs[path][1].items() if v is not None}, comp_rows)
        if not SKIP_PROFILE and cfg["name"] not in profiled:
            profiled[cfg["name"]] = profile_bundle(b, h, w, device)
        for k, v in summary["models"].items():
            if k.startswith(cfg["name"]):
                print(f"  {k}: mAP={v['mAP']:.4f} mIoU_all={v['mIoU_all']:.4f} "
                      f"mIoU_no_void={v['mIoU_no_void']:.4f} bpp={v.get('BPP', float('nan'))}")
        del b
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- codecs [C4] ----
    if not SKIP_CODECS:
        hm = (args.hevc_encoder_cfg_main, args.hevc_encoder_cfg_per_sequence,
              args.tapp_encoder_path, args.tapp_decoder_path)
        tasks = []
        for i, inputs in enumerate(make_loader(ds, args)):
            x4 = tuple_images(inputs, torch.device("cpu"))
            # 8-bit levels as float tensors, detached from any CUDA state
            tasks.append((keys[i], [x4[j].clone().contiguous() for j in range(N_CAM)], codec_dir, hm, list(CAMS)))
        codec_metrics = {}
        # 'spawn': forking after CUDA and the PyTorch thread pool are initialised deadlocks the
        # workers (seen as a hang at 0% in the first v2 run) [C12]
        n_workers = int(os.environ.get("EVAL_CODEC_WORKERS", max(1, (os.cpu_count() or 2) // 2)))
        if n_workers <= 1:
            _codec_worker_init()
            for t in tqdm(tasks, desc=f"{split_name} codecs (serial)"):
                key, m = codec_worker(t)
                codec_metrics[key] = m
        else:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(n_workers, initializer=_codec_worker_init) as pool:
                for key, m in tqdm(pool.imap_unordered(codec_worker, tasks), total=len(tasks),
                                   desc=f"{split_name} codecs ({n_workers} workers)"):
                    codec_metrics[key] = m
        base = load_bundle(DL_MODEL_CONFIGS[0], args, device)
        if BN_RECALIB_SCOPE == "all" and BN_RECALIB_N:
            bn_recalibrate(base, args, device, BN_RECALIB_N, BN_RECALIB_SEED)
        def codec_perception(bundle, name):
            det, seg = {}, {}
            for i, inputs in enumerate(tqdm(make_loader(ds, args), desc=f"{split_name} {name}")):
                key = keys[i]
                pngs = [os.path.join(codec_dir, name, f"{key}_{c}.png") for c in CAMS]
                if not all(os.path.exists(p) for p in pngs):
                    continue
                x_dec = torch.stack([transforms.ToTensor()(Image.open(p).convert("RGB")) for p in pngs]).to(device)
                d, c = perceive(bundle, features(bundle, x_dec), tuple_targets(inputs, img_size),
                                tuple_semantic(inputs, device), img_size, args)
                det[key] = d
                if c is not None:
                    seg[key] = c
            res = summarize(det, seg)
            for m in ("BPP", "PSNR", "MS_SSIM"):
                res[m] = float(np.nanmean([codec_metrics[k][name][m] for k in codec_metrics]))
            return res, det, seg

        perc2 = None
        if CODEC_PERC_JOINT:                             # [C22] same traditional pipeline, retrained perception
            perc2 = load_joint_bundle(dict(name=CODEC_PERC_JOINT, kind="joint", tag=CODEC_PERC_JOINT), args, device)
            if perc2 is None:
                print(f"  [codec perception] JOINT/{CODEC_PERC_JOINT} not found; plain baseline only")
        for name in list(HEVC_QPS) + list(J2K_QS):
            res, det, seg = codec_perception(base, name)
            summary["codecs"][name] = res
            dump(per_img_dir, "CODEC_" + name.replace(" ", "_"), det, seg,
                 {k: codec_metrics[k][name] for k in codec_metrics})
            if perc2 is not None:
                res, det, seg = codec_perception(perc2, name)
                res["perception"] = f"JOINT_{CODEC_PERC_JOINT}"
                summary["codecs"][f"{name} ({CODEC_PERC_JOINT})"] = res
                dump(per_img_dir, "CODEC_" + name.replace(" ", "_") + "_" + CODEC_PERC_JOINT, det, seg,
                     {k: codec_metrics[k][name] for k in codec_metrics})
            # [C15] the baseline fine-tuned on frames decoded at this QP (fairness control)
            if CODEC_FT and name in HEVC_QPS:
                adapted, msg = load_codec_ft(HEVC_QPS[name], args, device, base)
                if adapted is None:
                    print(f"  [no adapted baseline] {name}: {msg}")
                    continue
                res, det, seg = codec_perception(adapted, name)
                res["receiver"] = msg
                summary["codecs"][name + " (adapted)"] = res
                dump(per_img_dir, "CODEC_" + name.replace(" ", "_") + "_adapted", det, seg,
                     {k: codec_metrics[k][name] for k in codec_metrics})
                del adapted
    return summary


def run_meta(args, device):
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "git_commit": sh(["git", "rev-parse", "HEAD"]), "script": os.path.abspath(__file__),
        "flags": {"SPLITS": SPLITS, "BN_RECALIB_N_tuples": BN_RECALIB_N, "BN_RECALIB_SCOPE": BN_RECALIB_SCOPE,
                  "BN_RECALIB_SEED": BN_RECALIB_SEED, "PERCEPTION_INPUT": PERCEPTION_INPUT, "RX_BN": RX_BN, "RX_FT": RX_FT, "CODEC_FT": CODEC_FT, "RX_FT_SUFFIX": RX_FT_SUFFIX, "SX_SLIM": SX_SLIM, "SX_SUFFIX": SX_SUFFIX, "E2E": E2E, "JOINT_RUNS": JOINT_RUNS, "SENDER_ONLY": SENDER_ONLY,
                  "CODEC_PERC_JOINT": CODEC_PERC_JOINT, "HEVC_QPS": list(HEVC_QPS.values()), "E2E_SUFFIX": E2E_SUFFIX, "CODECS": os.environ.get("EVAL_CODECS", "all"),
                  "SKIP_CODECS": SKIP_CODECS, "VOID_INDEX": VOID_INDEX},
        "input_hw": [int(args.input_height), int(args.input_width)], "cameras": CAMS,
        "bpp_definition": "bits / (input_height x input_width) of one camera image, averaged over the four cameras",
        "detection_conf_thres": args.detection_conf_thres, "detection_nms_thres": args.detection_nms_thres,
        "loader_settings": {k: getattr(args, k, None) for k in ("dataset", "crop", "input_height", "input_width",
                                                                "dataset_dir", "train_file", "val_file")},
        "models": {c["name"]: f"{c['pretrained_path_segment']}/weights_{c['batch_suffix']}" for c in DL_MODEL_CONFIGS},
    }


if __name__ == "__main__":
    config = collect_args()
    args = Tupperware(yaml.safe_load(open(config.config)))
    # The loader builds its keys from these; force the SynWoodScape multi-camera setting [C7].
    args.dataset = "synwoodscape_raw"
    args.train = "compression_semantic_detection"
    args.batch_size = 1
    args.frame_idxs = [0]
    args.has_detection_annotations = True
    args.has_semantic_annotations = True
    hm_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    args.hevc_encoder_cfg_main = os.path.join(hm_base, "HM", "cfg", "encoder_intra_main.cfg")
    args.hevc_encoder_cfg_per_sequence = os.path.join(hm_base, "HM", "cfg", "per-sequence", "Traffic_Fisheye8K.cfg")
    args.tapp_encoder_path = os.path.join(hm_base, "HM", "bin", "TAppEncoderStatic")
    args.tapp_decoder_path = os.path.join(hm_base, "HM", "bin", "TAppDecoderStatic")
    args.pretrained_weights = os.environ.get(
        "EVAL_WEIGHTS", DEFAULT_WEIGHTS)
    if not os.path.isdir(args.pretrained_weights):
        sys.exit(f"Checkpoint root not found: {args.pretrained_weights}\nSet EVAL_WEIGHTS to the MTL_bitrates folder.")
    lists = {"val": os.environ.get("EVAL_VAL_FILE", args.val_file),
             "test": os.environ.get("EVAL_TEST_FILE", "data/test.txt")}
    # Every list must exist on disk in full, so a run never stops halfway [C11].
    check_lists(args, {**{s: lists[s] for s in SPLITS}, "bn_recalibration": BN_FILE or args.train_file})
    CAMS = None  # set from the dataset below
    probe = SynWoodScapeRawDataset(data_path=args.dataset_dir, path_file=lists[SPLITS[0]], is_train=False, config=args)
    CAMS = list(probe.all_cam_sides)
    assert len(CAMS) == N_CAM, CAMS

    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag += f"_bnRecal{BN_RECALIB_N}{BN_RECALIB_SCOPE}s{BN_RECALIB_SEED}_perc{PERCEPTION_INPUT}" if BN_RECALIB_N else ""
    tag += f"_rxBN{RX_BN}" + (f"_rxFT{RX_FT_SUFFIX}" if RX_FT else "") + ("_codecFT" if CODEC_FT else "") + (SX_SUFFIX.lower() if SX_SLIM else "") + (E2E_SUFFIX.lower() if E2E else "") + ("_joint" if JOINT_RUNS else "") + ("_senderonly" if SENDER_ONLY else "")
    out_root = os.path.join(args.output_directory, f"eval_svs_v2_{tag}")
    os.makedirs(out_root, exist_ok=True)
    shutil.copy(config.config, os.path.join(out_root, "config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = run_meta(args, device)
    meta["lists"] = {**{s: lists[s] for s in SPLITS}, "bn_recalibration": BN_FILE or args.train_file}
    with open(os.path.join(out_root, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    results, profiled = {"meta": meta}, {}
    for s in SPLITS:
        results[f"SynWoodScape_{s}"] = evaluate_split(args, s, lists[s], out_root, device, profiled)
        results["profile"] = profiled
        with open(os.path.join(out_root, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
    print(f"=> results in {out_root}")
