# Part C (SVS-JPC) re-evaluation: how to run

Inference only, no training. It regenerates Chapter 5's tables in the same way v4 did for Part B.

## 1. Put the files in place

Copy into the `eval/` folder of your `SVS_MTL` code:

- `evaluate_mic_mtl_v2.py` (this package)
- `stats_tests.py` (the one from the JointGAD `eval/` folder; it is unchanged)

Check `data/params.yaml`:

- `dataset_dir` must point at SynWoodScape (`SynWoodscape2021/`).
- `train_file: data/train.txt` is used for the BatchNorm statistics.
- `val_file: data/val.txt`

The script itself forces `dataset: synwoodscape_raw`, `train: compression_semantic_detection` and batch size 1, so you don't need to change the `woodscape_raw` / `virtual_multi_view` lines.

## 2. Run (from the SVS_MTL root)

Your local SynWoodScape holds frames 00000–00498. The `val.txt` and `train.txt` in the SVS zip were written for the full dataset (frames up to 08233), so use the three local lists in this package. Copy them to `data/`:

| List | Frames | What it is |
|---|---|---|
| `svs_val_local.txt` | 50 | The `val.txt` of the BEV repository |
| `svs_test_local.txt` | 50 | `test.txt`, the same in both repositories |
| `svs_bn_local.txt` | 399 | Frames 0–498 minus the 100 above, used only for BatchNorm statistics |

The 399 remaining frames match the ~400 steps per epoch in the checkpoint names. That suggests the models were trained on exactly those frames, with validation and test held out. Please confirm.

```bash
EVAL_VAL_FILE=data/svs_val_local.txt EVAL_TEST_FILE=data/svs_test_local.txt EVAL_BN_FILE=data/svs_bn_local.txt \
EVAL_WEIGHTS=/path/to/MTL_bitrates/ python eval/evaluate_mic_mtl_v2.py --config data/params.yaml 2>&1 | tee svs_eval_main.log
```

The codec stage now starts its workers with `spawn` instead of `fork`; the first run hung there because forking after CUDA is in use deadlocks. If it still stalls, `EVAL_CODEC_WORKERS=1` runs the codecs one at a time in the main process (slower but simple). To get the learned-model results without waiting for the codecs, add `EVAL_SKIP_CODECS=1`.

Before any model runs, the script now checks that all four camera images of every listed frame exist. If one is missing it stops and says which.

It evaluates the baseline and the four λ models on the validation and test lists. For each split it writes:

- `results.json`: per model, two rows: `__S` (sender side) and `__R` (receiver side). Codec rows run the baseline on HEVC- and JPEG2000-decoded images.
- `per_image/*.pkl`: per-tuple records for the statistics.
- `run_meta.json`: GPU, flags, checkpoints and lists.

Things to look at in the first minutes of the log:

- **Every model prints its BatchNorm mode.** For an encoder with no running statistics, the script prints `recalibrated on 250 training tuples`.
- **If a module fails to load, the run stops** and lists the missing or unexpected keys. Please send me that message rather than bypassing it; the old script silently continued with random weights in that case.

## 3. Statistics

```bash
python eval/stats_tests.py <run>/SynWoodScape_val/per_image  --ref SVS_Baseline_no_compression --csv stats_val.csv
python eval/stats_tests.py <run>/SynWoodScape_test/per_image --ref SVS_Baseline_no_compression --csv stats_test.csv
```

The resampling unit is the four-camera tuple.

## 4. One sensitivity run (same as Part B)

```bash
EVAL_BN_RECALIB_SCOPE=all EVAL_SKIP_CODECS=1 EVAL_VAL_FILE=data/svs_val_local.txt EVAL_TEST_FILE=data/svs_test_local.txt EVAL_BN_FILE=data/svs_bn_local.txt EVAL_WEIGHTS=... python eval/evaluate_mic_mtl_v2.py --config data/params.yaml
```

This also re-estimates the baseline's statistics, so the comparison is like-for-like.

## 5. Send back

- Both run folders, minus `codec_decoded/`, which is large.
- `stats_val.csv` and `stats_test.csv`
- The console logs

Please also answer these three questions; the thesis needs them:

1. **Training data.** Were the four `res50_LDMIC…_SVS` checkpoints and `res50_no_mic_no_compression…_SVS` trained on SynWoodScape? The `params.yaml` in the zip is set to WoodScape virtual multi-view.
2. **Steps per epoch.** The checkpoint names imply about 400 steps per epoch, but `train.txt` lists 8,000 tuples and the batch size is 1. Which training list or batch size was used?
3. **Source of the published numbers.** Which script and run produced the published Chapter 5 numbers (58.29 / 63.58; VVC; no-fusion; the ablations)?

## What changed from evaluate_mic_mtl.py

- **[C1]** Sender-side and receiver-side perception for every SVS-JPC model.
- **[C2]** BatchNorm runs as trained, with statistics re-estimated where the checkpoint has none. The old script forced per-image statistics on every encoder, the baseline included.
- **[C3]** Strict loading of every module.
- **[C4]** Codecs get perception through the baseline.
- **[C5]** Rate per pixel of one camera image, averaged over the cameras, with the per-camera rates as well.
- **[C6]** The duplicate "fusion-only" entry is dropped.
- **[C7]** SynWoodScape only. The old script also evaluated random WoodScape image sets on the training list.
- **[C8]** Empty images count as false positives; mIoU is reported with and without void; 8-bit images are used for PSNR / MS-SSIM.
- **[C9]** Per-tuple pickles that `stats_tests.py` can read.
- **[C10]** `run_meta.json`.

I checked the script against the released model code with random weights. Loading, the encoder, fusion and codec passes, both perception paths, the summaries and the profiling all ran. The data loader and the HM/JPEG2000 calls could not be run here, because they need SynWoodScape and HM.

Size of the released model, from instantiating it at 544×288:

| Part | Parameters | GMACs |
|---|---|---|
| ResNet-50 encoder | 23.5 M (+2.0 M unused classifier) | 51.6 (four images) |
| Fusion on F4 | **351.3 M** | **215.0** |
| Codec | 20.3 M | 338.6 |
| Detection head | 21.6 M | – |
| Segmentation head | 9.5 M | – |
| **Total** | **≈ 428 M** | |

This is not the 103 M stated in the thesis.

## 6. Receiver-side BatchNorm run (Part B protocol)

The first full run used the BatchNorm statistics saved in the checkpoints for every encoder pass. Those were estimated on original images, so on the decoded images x_hat the receiver pass normalizes inputs it never saw. Part B's receiver-side parity appeared only after re-estimating these statistics on passes over x_hat. To give Part C the same treatment:

```bash
EVAL_RX_BN=xhat EVAL_SKIP_CODECS=1 \
EVAL_VAL_FILE=data/svs_val_local.txt EVAL_TEST_FILE=data/svs_test_local.txt EVAL_BN_FILE=data/svs_bn_local.txt \
EVAL_WEIGHTS=/media/PorableSSD/Work/Academic/PHDWork/development/MTL/WoodScapeGit/omnidet/pretrained_models/MyModel/MTL_bitrates/ \
python eval/evaluate_mic_mtl_v2.py --config data/params.yaml
```

What this run does:

- **Only the receiver-side encoder pass changes.** It uses a copy of the encoder whose statistics come from reconstructions of 250 training tuples (the `svs_bn_local.txt` frames). No weight changes.
- **The sender encoder that makes the bitstream is untouched**, so `__S` and the rates come out identical to the first run.
- **The codecs are skipped**, because their rows are unchanged (the baseline keeps its own statistics). This makes the comparison slightly favour SVS-JPC: its receiver is adapted to its own reconstructions, and the baseline is not adapted to HEVC.

Then run `stats_tests.py` on both runs' `per_image` folders and send me the CSVs and the new `results.json`.

## 7. Receiver fine-tuning (option 1)

The sender (encoder, F3 projection, codec) stays frozen, so the bitstream and rates are those of the published checkpoints. The script fine-tunes a receiver copy of the encoder, the fusion module and both heads on the reconstructions of the 399 training frames in `svs_bn_local.txt`.

- **Same losses as the SVS trainer:** YOLO detection loss, focal segmentation loss with ENet weights, uncertainty weighting.
- **BatchNorm statistics:** first re-estimated on x_hat, then kept fixed.
- **No checkpoint selection:** it trains a fixed number of epochs and saves the last one, so validation and test stay untouched.

```bash
cp finetune_svs_receiver.py eval/
EVAL_WEIGHTS=/media/PorableSSD/Work/Academic/PHDWork/development/MTL/WoodScapeGit/omnidet/pretrained_models/MyModel/MTL_bitrates/ \
FT_TRAIN_FILE=data/svs_bn_local.txt FT_EPOCHS=15 FT_LR=5e-5 \
python eval/finetune_svs_receiver.py --config data/params.yaml 2>&1 | tee svs_rxft.log
```

- **It runs all four λ in turn.** Use `FT_LAMBDAS=0.01` to try one first; that's the most useful check of whether it works at all.
- **Output:** each run writes `<run>_RXFT/models/weights_15/` next to the original checkpoint folder, including `rxft_meta.json` with the per-epoch losses.
- **If GPU memory runs out**, set `FT_TRAIN_FUSION=0`. That freezes the 351 M-parameter fusion module and trains only the encoder and heads (57 M parameters).

Then evaluate the receiver with the fine-tuned weights. The codecs are not needed again:

```bash
EVAL_RX_FT=1 EVAL_SKIP_CODECS=1 \
EVAL_VAL_FILE=data/svs_val_local.txt EVAL_TEST_FILE=data/svs_test_local.txt EVAL_BN_FILE=data/svs_bn_local.txt \
EVAL_WEIGHTS=... python eval/evaluate_mic_mtl_v2.py --config data/params.yaml
```

The `__S` rows and the bpp must be identical to the earlier runs; only `__R` changes. Please send `results.json`, `svs_rxft.log` and one `rxft_meta.json`.

## 8. Task-aware codec, end to end (option 2)

Options 1 and 3 kept the published bitstream. Here the codec itself learns from perception:

- **Trained:** the F3 projection, the entropy model and the joint decoder, with their original loss (λ·255²·MSE + rate), plus the detection and segmentation losses of a receiver that perceives the reconstructions. The receiver (slim fusion) is trained jointly and starts from `_RXFT_SLIM`. All three losses get uncertainty weights, as in the original training. As in the original training, the likelihoods use uniform noise and the decoder gets the rounded latent (straight-through gradient).
- **Frozen:** the sender encoder, so on-board (`__S`) perception is unchanged.
- **Learning rates:** `FT_LR_CODEC=1e-5` for the codec, `FT_LR` for the receiver.

```bash
bash run_svs_steps.sh e2e1       # about as long as ft1_slim plus the codec backward pass
bash run_svs_steps.sh eval_e2e   # re-measures bpp and PSNR; HEVC plain and adapted in the same run
bash run_svs_steps.sh stats
bash run_svs_steps.sh pack
```

The rate is re-measured, so it may move away from 0.12 bpp. If it leaves the QP 39–42 bracket, the adapted HEVC rows need another QP, for example `FT_HEVC_QPS=36 bash run_svs_steps.sh hevc_ft`, then `eval_e2e` again.

The log prints bpp and PSNR per epoch. If bpp climbs steadily, the task losses are buying accuracy with bits; send me the log either way.

## 9. Geometry-aware attention (`geo` step)

Each camera attends only to its two overlapping neighbours: FV to MVL and MVR, RV to MVL and MVR, and each mirror camera to FV and RV. Every feature position carries an encoding of the direction it looks in, in the vehicle frame. That direction comes from a nominal equidistant fisheye model (the image width spans 190°) and the nominal mounting yaw of each camera (0°, ±90°, 180°), plus a small learned correction. No calibration file is used. Positions in the overlap zones look in the same direction, so they can find each other. Each block starts as the identity, so a run starts exactly where the trained model is.

- **Inside the LDMIC decoder** (`FT_CODEC_GEO=1`): one block after each of the two joint context transfers, 0.27 M parameters in total. It is trained with the task-aware codec run; everything else is identical to `_E2E`, so the comparison isolates the new blocks.
- **On the sender's F4 fusion** (`FT_FUSION=geo`): 1.4 M parameters, against 6.5 M for the slim fusion and 351 M for the released one.
- **Control** (`FT_FUSION=none`): the sender heads retrained without any fusion. This tells whether fusion helps at all.

```bash
bash eval/run_svs_steps.sh geo      # waits for a running curve step, then about 1 h
tail -f logs/geo.log
```

## 10. Latent skip (`yskip` step)

The transmitted latent ŷ is a projection of the sender's own F3 features. Until now the receiver used it only to decode pixels and then re-encoded those pixels. With the latent skip, the receiver adds ŷ to its own F3 of x̂ through a small adapter that starts at zero, then computes F4 from the sum. It perceives from the pixels and from the transmitted features, at the same rate.

- **Receiver only** (`_RXFT_SLIM_YSKIP`): codec frozen, so the bitstream is unchanged. Compare with R3.
- **Task-aware codec with the skip** (`_E2E_YSKIP`): the detection and segmentation gradients also reach the projection through ŷ, so the codec can learn to put task information into the bits. Compare with R4 and adapted HEVC.

```bash
bash eval/run_svs_steps.sh yskip      # waits for other runs, then about 1 h
tail -f logs/yskip.log
```

### Latent skip at higher rates (`yskip_hr`)

Same as `yskip`, but starting from the λ = 0.08 codec (0.27–0.29 bpp). It runs the task-aware codec with the skip at λ = 0.08 itself, and fine-tuned towards λ = 0.3 and 1.0 (more bits). Together with the λ = 0.01 run this gives three or four rate points. Start it after `yskip`; it waits for the other steps.

## 11. Overnight sender-side study (`overnight`)

**Framing.** One shared network on one SoC perceives on board and writes the bitstream. Nothing is perceived from the decoded images; there is no second pass. Every run trains the whole sender together, as in the original multi-task training: shared encoder, fusion, detection and segmentation heads, F3 projection and LDMIC codec. The loss is uncertainty-weighted detection, segmentation and λ·255²·MSE + rate. Training uses the 399 local tuples with augmentation, `ON_EPOCHS` epochs (default 40), and keeps the last checkpoint.

| Run | What it is |
|---|---|
| `BASE_FT` | Perception-only control: the uncompressed baseline trained on the same schedule, so extra training cannot pass for a compression gain |
| `SWEEP_L0.01` … `SWEEP_L2.0` | Slim fusion and the geometry-aware LDMIC decoder; λ 0.01, 0.03, 0.08, then 0.25, 0.6 and 2.0 from the λ = 0.08 checkpoint (roughly 0.1 → 1 bpp) |
| `ABL_NOFUSION`, `ABL_GEO`, `ABL_NOCGEO`, `ABL_HEAVY` | Ablations at λ = 0.03: no fusion, geometry-aware fusion, slim without the decoder geometry blocks, the released 351 M fusion |

**Evaluation (sender side only).**

- **Rate and fidelity of every bitstream:** bpp, PSNR, MS-SSIM.
- **On-board detection and segmentation.**
- **Traditional pipeline:** original image → HEVC (QP 24–45) or JPEG2000 → decoded → detection and segmentation, perceived by the baseline and by `BASE_FT`.
- **Statistics:** against the baseline and against `BASE_FT`.
- **Profile:** latency of perception plus writing the bitstream, on the same GPU.

```bash
bash eval/run_svs_steps.sh overnight          # waits for any running step, then about 8 h
tail -f logs/overnight.log
ON_EPOCHS=30 bash eval/run_svs_steps.sh overnight   # if it must finish sooner
```

A failed run is logged and skipped. A finished run is not retrained if the step is started again, so an interrupted night can be resumed with the same command.

## 12. Follow-up of the overnight study (`boost`)

- **`ABL_ADAPTER`:** a per-camera adapter the size of the slim fusion, with no exchange between cameras. If it matches the slim fusion, the 7–8 point gain over no fusion comes from the extra trainable capacity, not from the other cameras.
- **`SIDED_L*` (main):** the side branch with the codec loss stopped at F3, across λ 0.01–1.5, 30 epochs each, each a warm start from `SWEEP_L0.03`. The quick check at λ = 0.6 (15 epochs) kept on-board perception at 77.1 / 62.3 (test) and reached 31.3 dB at 1.13 bpp.

```bash
bash eval/run_svs_steps.sh boost      # waits for any running step, then about 5 h
tail -f logs/boost.log
BO_EPOCHS=15 bash eval/run_svs_steps.sh boost   # shorter
```

## 13. Full study of the final design (`full`)

**Final design:**
- Start from the released λ = 0.03 checkpoint.
- Slim fusion.
- The side image-analysis branch added to the F3 projection.
- The codec loss stops at F3, so the shared encoder is trained by perception only and the codec branch sets the rate freely.
- No geometry blocks in the decoder (they had no effect).

**Runs** (all with `FU_EPOCHS` epochs, default 30):

| Group | Runs |
|---|---|
| Rate sweep | `FULL_L0.01`, `0.03`, `0.1`, `0.3`, `0.6`, `1.5` (about 0.15–1.3 bpp) |
| Ablation at λ = 0.1: fusion | `FULL_A_NOFUSION`, `FULL_A_ADAPTER` (same capacity, no exchange between cameras), `FULL_A_GEO`, `FULL_A_HEAVY` (released 351 M fusion) |
| Ablation at λ = 0.1: design | `FULL_A_NOSIDE` (no side branch), `FULL_A_SHARED` (codec loss reaches the encoder) |

**Evaluation** (sender side only):

- **Models compared:** the released models, `BASE_FT` and all `FULL` runs.
- **Codecs:** HEVC at QP 22–45 and JPEG2000, perceived by the baseline and by `BASE_FT`.
- **Also reported:** statistics against both, per-class AP, and timing of perception plus bitstream writing.

```bash
bash eval/run_svs_steps.sh full          # waits for other runs, then about 7-8 h; resumable
tail -f logs/full.log
FU_EPOCHS=20 bash eval/run_svs_steps.sh full     # shorter
```
