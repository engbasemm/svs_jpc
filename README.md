# SVS-JPC / SVS-JPC+ — thesis evaluation and training code (Part C, Chapter 5)

One network on one SoC perceives four surround-view fisheye cameras on board (detection +
segmentation) and, in the same pass, writes a learned bitstream for other ECUs, a phone or the
cloud. From the PhD thesis *Task-Aware Learned Compression and Perception in Automotive Fisheye
Camera Systems* (B. A. H. Barakat, Cairo University).

It plugs into the SVS_MTL / OmniDet code base (`omnidet/`), whose models and SynWoodScape loader it imports.

## Layout

| Path | Content |
|---|---|
| `eval/evaluate_mic_mtl_v2.py` | evaluation: on-board (sender) perception, bitstream quality, HEVC (HM) / JPEG2000 chains, statistics hooks, GMACs and latency |
| `eval/finetune_svs_receiver.py` | training: joint runs (`FT_MODE=joint`), final design SVS-JPC+ (`FT_SIDE=1 FT_DETACH_CODEC=1`), fusion variants (`FT_FUSION=slim/geo/adapter/heavy/none`), receiver studies |
| `eval/stats_tests.py` | paired bootstrap CIs and per-image tests |
| `scripts/run_svs_steps.sh` | every experiment as a named step (`full`, `overnight`, `quick`, ..., `stop`, `restart`, `stats`, `pack`) |
| `scripts/run_svs_steps_low.sh` | low-rate end of SVS-JPC+ (λ 0.003, 0.001) |
| `data/svs_{val,test,bn}_local.txt` | 50 validation / 50 test / 399 training tuples (local SynWoodScape frames 0-498) |
| `results/<experiment>/` | `results*.json`, statistics CSVs and run metadata of every experiment (E01-E12 of the thesis workbook) |
| `docs/RUN_SVS_EVAL.md` | step-by-step run guide |

## Run

```bash
cd omnidet
cp <this repo>/eval/*.py eval/ && cp <this repo>/data/*.txt data/ && cp <this repo>/scripts/*.sh .
bash run_svs_steps.sh full        # SVS-JPC+ rate sweep + ablations + evaluation + stats + pack (~6 h, background)
tail -f logs/full.log
bash run_svs_steps.sh stop        # stop everything safely
```

## Main result (SVS-JPC+, test tuples, `results/full/results_full.json` and `results/full_low/results_full_low.json`)

| λ | bpp | PSNR | Δ vs JPEG2000 | MS-SSIM Δ vs HEVC | On-board mAP / mIoU | ΔmAP vs perception-only (same schedule) |
|---|---|---|---|---|---|---|
| 0.001 | 0.056 | 26.24 | – | +0.037 (PSNR +0.4 dB vs HEVC) | 79.2 / 62.5 | +4.5 (sig.) |
| 0.003 | 0.117 | 27.24 | +3.2 dB | +0.015 | 79.3 / 62.5 | +4.6 (sig.) |
| 0.01 | 0.267 | 29.25 | +2.8 dB | +0.006 | 78.8 / 62.4 | +4.1 (sig.) |
| 0.03 | 0.497 | 30.83 | +2.4 dB | +0.000 | 79.1 / 62.4 | +4.4 (sig.) |
| 0.1 | 0.810 | 31.60 | +1.3 dB | −0.004 | 78.6 / 62.3 | +3.9 (sig.) |
| 0.3 | 1.073 | 31.93 | +0.4 dB | −0.006 | 78.5 / 62.4 | +3.8 (sig.) |

Uncompressed baseline 76.9 / 65.4; perception-only network on the same schedule (BASE_FT) 74.7 / 65.4.
Sender latency (RTX 4070 Ti Super, four cameras): 17.1 ms with the bitstream vs 13.0 ms perception only.

## Not in this repository
Checkpoints and SynWoodScape (licence); see the thesis for their sources.
