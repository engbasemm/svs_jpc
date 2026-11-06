"""
stats_tests.py - significance tests for every model against the baseline (and any pair you name).

Reads the per-image pickles that evaluate_mtl_v4.py writes (<run>/Woodscape/per_image/*.pkl).
No network is run. Replaces perform_ttest.py.

For each model vs the reference (default: the uncompressed baseline) it reports:
  1. Dataset-level metrics (mAP, mIoU with / without void) with a PAIRED BOOTSTRAP 95% CI of
     the difference. mAP is a dataset-level statistic, so this is the primary test for it.
  2. Per-image metrics (detection AP, detection F1, mIoU with / without void) with a
     PAIRED t-test, a Wilcoxon signed-rank test (no normality assumption), the effect size
     Cohen's d_z, and Holm-corrected p-values across all models compared.

Usage (from omnidet/, so train_utils resolves):
  python stats_tests.py <run>/Woodscape/per_image [--ref RES50_Baseline_no_compression]
                        [--n 2000] [--block 0] [--void 0] [--seed 0] [--csv out.csv]
--block N > 0 resamples blocks of N consecutive frame numbers (neighbouring frames are
correlated); use e.g. 10 as a sensitivity check. The t-test assumes independent images; report
the block-bootstrap result next to it.
"""
import argparse
import csv
import glob
import math
import os
import pickle
import re

import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # omnidet/
from train_utils.detection_utils import ap_per_class


def to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def load(path):
    rec = pickle.load(open(path, "rb"))
    for d in rec.get("det", {}).values():
        if d is not None:
            d["stats"] = [[to_np(a) for a in s] for s in d["stats"]]
    return rec


# ---------------- dataset-level ----------------
def dataset_metrics(rec, keys, void):
    st, lab, conf = [], [], None
    for k in keys:
        d = rec["det"].get(k)
        if d is not None:
            st.extend(d["stats"])
            lab.extend(d["target_cls"])
        c = rec["seg"].get(k)
        if c is not None:
            conf = c.astype(np.float64) if conf is None else conf + c
    out = {"mAP": np.nan, "mIoU_all": np.nan, "mIoU_no_void": np.nan}
    if st and lab:
        tp, sc, pc = [np.concatenate(x, 0) for x in zip(*st)]
        out["mAP"] = float(np.mean(ap_per_class(tp, sc, pc, np.array(lab))[2]))
    if conf is not None:
        out.update(iou_summary(conf, void))
    return out


def iou_summary(conf, void, present_only=False):
    tp = np.diag(conf).astype(np.float64)
    denom = conf.sum(0) + conf.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / denom
    if present_only:  # per image: only classes in the ground truth or the prediction
        iou = np.where(denom > 0, iou, np.nan)
    return {"mIoU_all": float(np.nanmean(iou)), "mIoU_no_void": float(np.nanmean(np.delete(iou, void)))}


# ---------------- per-image ----------------
def per_image(rec, key, void):
    out = {"AP_img": np.nan, "F1_img": np.nan, "mIoU_all_img": np.nan, "mIoU_no_void_img": np.nan}
    d = rec["det"].get(key)
    if d is not None and d["target_cls"]:
        n_gt = len(d["target_cls"])
        if d["stats"]:
            tp, sc, pc = [np.concatenate(x, 0) for x in zip(*d["stats"])]
            out["AP_img"] = float(np.mean(ap_per_class(tp, sc, pc, np.array(d["target_cls"]))[2])) if len(tp) else 0.0
            n_tp = float(tp.sum()) if tp.ndim == 1 else float(tp[:, 0].sum())
            n_pred = len(tp)
        else:
            out["AP_img"], n_tp, n_pred = 0.0, 0.0, 0
        prec = n_tp / n_pred if n_pred else 0.0
        rec_ = n_tp / n_gt
        out["F1_img"] = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0
    c = rec["seg"].get(key)
    if c is not None:
        s = iou_summary(c, void, present_only=True)
        out["mIoU_all_img"], out["mIoU_no_void_img"] = s["mIoU_all"], s["mIoU_no_void"]
    return out


def groups(keys, block):
    if block <= 0:
        return [[k] for k in keys]
    g = {}
    for k in keys:
        m = re.match(r"(\d+)", k)
        g.setdefault(int(m.group(1)) // block if m else k, []).append(k)
    return list(g.values())


def cam_filter(keys, cams):
    if not cams:
        return keys
    want = {c.strip().upper() for c in cams.split(",")}
    return [k for k in keys if os.path.splitext(k)[0].split("_")[-1].upper() in want]


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_image_dir")
    ap.add_argument("--ref", default="RES50_Baseline_no_compression")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--void", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--cams", default="", help="only frames of these cameras, e.g. FV,RV (from the file name suffix)")
    a = ap.parse_args()

    ref = load(os.path.join(a.per_image_dir, a.ref + ".pkl"))
    others = [p for p in sorted(glob.glob(os.path.join(a.per_image_dir, "*.pkl")))
              if os.path.basename(p)[:-4] != a.ref]
    rng = np.random.default_rng(a.seed)
    rows = []

    print(f"Reference: {a.ref}   bootstrap n={a.n} block={a.block}   cameras={a.cams or 'all'}   (diff = model - reference)\n")
    print("1) DATASET-LEVEL, paired bootstrap 95% CI")
    print(f"{'model':<30}{'metric':<14}{'model':>8}{'ref':>8}{'diff':>9}{'2.5%':>9}{'97.5%':>9}  sig")
    for path in others:
        name = os.path.basename(path)[:-4]
        rec = load(path)
        keys = cam_filter(sorted((set(rec["det"]) | set(rec["seg"])) & (set(ref["det"]) | set(ref["seg"]))), a.cams)
        grp = groups(keys, a.block)
        pm, pr = dataset_metrics(rec, keys, a.void), dataset_metrics(ref, keys, a.void)
        boots = {m: [] for m in pm}
        for _ in range(a.n):
            ks = [k for i in rng.integers(0, len(grp), len(grp)) for k in grp[i]]
            bm, br = dataset_metrics(rec, ks, a.void), dataset_metrics(ref, ks, a.void)
            for m in boots:
                boots[m].append(bm[m] - br[m])
        for m in pm:
            d = np.array(boots[m], dtype=float)
            if np.all(np.isnan(d)):
                continue
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            sig = "yes" if lo > 0 or hi < 0 else "no"
            print(f"{name:<30}{m:<14}{pm[m]:>8.4f}{pr[m]:>8.4f}{pm[m]-pr[m]:>+9.4f}{lo:>+9.4f}{hi:>+9.4f}  {sig}")
            rows.append(dict(model=name, level="dataset", metric=m, model_value=pm[m], ref_value=pr[m],
                             diff=pm[m] - pr[m], ci_low=lo, ci_high=hi, n=len(keys), block=a.block,
                             t=np.nan, p_t=np.nan, p_wilcoxon=np.nan, cohen_dz=np.nan, p_t_holm=np.nan,
                             significant=sig))

    print("\n2) PER-IMAGE, paired t-test / Wilcoxon / Cohen's d_z, Holm-corrected across models")
    metrics = ["AP_img", "F1_img", "mIoU_all_img", "mIoU_no_void_img"]
    per_metric = {m: [] for m in metrics}
    for path in others:
        name = os.path.basename(path)[:-4]
        rec = load(path)
        keys = cam_filter(sorted((set(rec["det"]) | set(rec["seg"])) & (set(ref["det"]) | set(ref["seg"]))), a.cams)
        vm = [per_image(rec, k, a.void) for k in keys]
        vr = [per_image(ref, k, a.void) for k in keys]
        for m in metrics:
            x = np.array([v[m] for v in vm], float)
            y = np.array([v[m] for v in vr], float)
            ok = ~(np.isnan(x) | np.isnan(y))
            x, y = x[ok], y[ok]
            if len(x) < 3:
                continue
            d = x - y
            t, p_t = stats.ttest_rel(x, y)
            try:
                p_w = stats.wilcoxon(x, y, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p_w = np.nan
            dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
            per_metric[m].append(dict(model=name, metric=m, mean_model=x.mean(), mean_ref=y.mean(),
                                      diff=d.mean(), t=t, p_t=p_t, p_w=p_w, dz=dz, n=len(x)))
    print(f"{'model':<30}{'metric':<18}{'model':>8}{'ref':>8}{'diff':>9}{'t':>8}{'p(t)':>9}"
          f"{'p(Holm)':>9}{'p(Wilc)':>9}{'d_z':>7}{'n':>5}")
    for m in metrics:
        res = per_metric[m]
        if not res:
            continue
        adj = holm(np.array([r["p_t"] for r in res]))
        for r, pa in zip(res, adj):
            print(f"{r['model']:<30}{m:<18}{r['mean_model']:>8.4f}{r['mean_ref']:>8.4f}{r['diff']:>+9.4f}"
                  f"{r['t']:>8.2f}{r['p_t']:>9.4f}{pa:>9.4f}{r['p_w']:>9.4f}{r['dz']:>7.2f}{r['n']:>5d}")
            rows.append(dict(model=r["model"], level="per_image", metric=m, model_value=r["mean_model"],
                             ref_value=r["mean_ref"], diff=r["diff"], ci_low=np.nan, ci_high=np.nan,
                             n=r["n"], block=0, t=r["t"], p_t=r["p_t"], p_wilcoxon=r["p_w"],
                             cohen_dz=r["dz"], p_t_holm=pa, significant="yes" if pa < 0.05 else "no"))
    print("\nNotes: mAP is not an average of per-image values, so its test is the bootstrap in (1);"
          "\nper-image AP/F1 are complementary. Frames from the same sequence are correlated; "
          "re-run with --block 10.")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"=> {a.csv}")


if __name__ == "__main__":
    main()
