#!/usr/bin/env bash
# Part C (SVS-JPC) runs, one step at a time.  Run from the omnidet root:
#   cd /media/basem/Collection/Work/Academic/PHDWork/PHD_Topics/MTL/ValeoMTL/SynWoodScapePrj/omnidet
#   bash run_svs_steps.sh <step>
#
# Steps:
#   setup <folder>  copy the scripts and lists from the unzipped package <folder> into eval/ and data/
#   ft1             option 1: fine-tune the receiver (heavy fusion) for lambda 0.01
#   eval_ft         evaluate the option-1 receivers (codecs skipped)
#   ft_rest         option 1 for lambda 0.08, 0.05, 0.03
#
#   Options 1 + 3 (slim fusion, ~6.5 M parameters instead of 351 M), lambda 0.01 first:
#   ft1_slim        receiver with slim fusion, fine-tuned on x_hat               -> <run>_RXFT_SLIM
#   sx1_slim        sender-side slim fusion + heads (sender encoder frozen)      -> <run>_SXSLIM
#   eval_slim       evaluate both (sender path slim, receiver path slim; codecs skipped)
#
#   Fairness control for the HEVC rows:
#   hevc_ft         baseline fine-tuned on HEVC-decoded training frames at QP 39 and 42 (brackets 0.12 bpp)
#   eval_hevc_ft    HEVC rows (plain and adapted) + the slim receiver, same run
#
#   Option 2 (task-aware codec, end to end), lambda 0.01 first:
#   e2e1            projection + codec + slim receiver trained together with the perception losses
#                   (sender encoder frozen; the bitstream changes)                -> <run>_E2E
#   eval_e2e        new rates + receiver, sender slim, HEVC plain and adapted (QP 39, 42), same run
#   e2e_all         everything for option 2 in one go: e2e1 (or waits for a running one, or skips it if
#                   already trained), then eval_e2e, stats, pack. Runs in the background; log in
#                   logs/e2e_all.log.  Check with:  tail -f logs/e2e_all.log
#
#   final           last checks, all in the background (log: logs/final.log):
#                   HEVC adapted at QP 36 (skipped if done), eval_e2e, stats against the baseline and
#                   against adapted HEVC QP39, pack.   Follow with:  tail -f logs/final.log
#   curve           overnight: the other three lambda (0.08, 0.05, 0.03) through the same pipeline as 0.01
#                   (slim receiver, slim sender, task-aware codec), HEVC adapted at QP 33, then eval_e2e,
#                   stats_vs, stats, pack. Background; log in logs/curve.log (about 2.5 h).
#   geo             geometry-aware attention (lambda 0.01), background, waits for a running curve step;
#                   log in logs/geo.log (about 1 h):
#                     - task-aware codec with geometry-aware attention inside the LDMIC decoder  -> _E2E_CGEO
#                       (same as _E2E otherwise, so the comparison isolates the new blocks)
#                     - sender with geometry-aware F4 fusion                                     -> _SXGEO
#                     - sender without any fusion (control)                                      -> _SXNOFUSION
#                   then both evaluations, statistics and pack.
#   highrate        higher rates: the lambda 0.08 codec fine-tuned end to end towards lambda 0.3 and 1.0
#                   (more bits), slim receiver; HEVC adapted at QP 31 and 33; evaluation, statistics
#                   against adapted HEVC QP33, pack. Background, waits for other runs; logs/highrate.log
#   yskip           latent skip (lambda 0.01): the receiver also reads the transmitted latent y_hat at F3.
#                     - receiver fine-tune, codec frozen (bitstream unchanged)      -> _RXFT_SLIM_YSKIP
#                     - task-aware codec with the latent skip (warm start from it)  -> _E2E_YSKIP
#                   then both evaluations (the second with HEVC), statistics, pack. Background, waits
#                   for other runs; log in logs/yskip.log (about 1 h).
#   yskip_hr        latent skip at higher rates: lambda 0.08 receiver with the skip, then the task-aware
#                   codec with the skip at lambda 0.08 (0.27-0.29 bpp), 0.3 and 1.0 (more bits);
#                   HEVC adapted at QP 31/33 if missing; evaluations, statistics, pack. Background,
#                   waits for the other steps; log in logs/yskip_hr.log (about 2 h).
#   overnight       SENDER-SIDE study, one network on one SoC, no second pass (about 8 h at 40 epochs):
#                   every run trains encoder, fusion, heads and codec together (FT_MODE=joint),
#                   FT_EPOCHS (default here 40) on the 399 local tuples, last checkpoint.
#                     1. BASE_FT   perception-only control: the baseline trained on the same schedule
#                     2. rate sweep, slim fusion + geometry-aware LDMIC decoder, lambda
#                        0.01, 0.03, 0.08, 0.25, 0.6, 2.0  (about 0.1 -> 1 bpp)
#                     3. ablation at lambda 0.03: no fusion, geometry-aware fusion, slim without the
#                        decoder geometry blocks, released (heavy) fusion
#                   then one evaluation (sender side only) with HEVC QP 24-45 and JPEG2000 perceived by
#                   the baseline and by BASE_FT, statistics against both, pack. Background; waits for
#                   other runs; log in logs/overnight.log. A failed run is logged and skipped.
#   boost           follow-up (about 4.5 h): the side branch with the codec loss stopped at F3 (SIDED, as in
#                   the quick check) across the rate range, plus the adapter control:
#                     1. ABL_ADAPTER   fusion-sized per-camera adapter WITHOUT cross-view exchange
#                                      (lambda 0.03, 40 epochs): does the gain come from the other cameras?
#                     2. SIDED_L*      side branch, codec loss stops at F3, warm start from SWEEP_L0.03,
#                                      lambda 0.01, 0.03, 0.1, 0.3, 0.6, 1.5, BO_EPOCHS (default 30) epochs
#                   then one sender-side evaluation with all runs, HEVC/JPEG2000, statistics, pack.
#                   Background; waits for other runs; log in logs/boost.log.
#   full            FULL STUDY of the final design (about 7-8 h at 30 epochs), all sender side:
#                   design = released lambda-0.03 model, slim fusion, side analysis branch, codec loss
#                   stopped at F3 (encoder trained by perception only), FU_EPOCHS (default 30) epochs.
#                     sweep     FULL_L<lambda>  lambda 0.6, 0.1, 0.03, 0.3, 0.01, 1.5   (~0.15 -> 1.3 bpp)
#                     ablation  at lambda 0.1:  FULL_A_NOFUSION, FULL_A_ADAPTER (no cross-view exchange),
#                               FULL_A_NOSIDE (no side branch), FULL_A_SHARED (codec loss reaches the
#                               encoder), FULL_A_GEO, FULL_A_HEAVY (released 351 M fusion)
#                   then one evaluation: released models, BASE_FT, all FULL runs, HEVC QP 22-45 and
#                   JPEG2000 perceived by the baseline and BASE_FT, statistics against both, profiling,
#                   pack. Background; waits for other runs; resumable; log in logs/full.log.
#   full_low        after full: two lower rates of SVS-JPC+ (lambda 0.003 and 0.001) for the 0.1-0.3 bpp end,
#                   evaluated with BASE_FT and FULL_L0.01 only (about 1.5 h). Log: logs/full_low.log
#   stop            stop every step and the training / evaluation / HM processes they started, in the safe
#                   order (steps first, so nothing new starts), and check the GPU is free
#   restart <step>  stop, then start <step> (e.g. bash eval/run_svs_steps.sh restart full)
#   stats_vs [ref] [run]  statistics against another row (default CODEC_HM_HEVC_QP39_adapted)
#
#   stats           paired bootstrap / tests on the newest evaluation run (val and test)
#   pack            put everything I need into one zip:  svs_results_<date>.zip
#
# Optional: FT_TRAIN_FUSION=0 bash run_svs_steps.sh ft1   if the GPU runs out of memory.
set -euo pipefail

export EVAL_WEIGHTS="/media/PorableSSD/Work/Academic/PHDWork/development/MTL/WoodScapeGit/omnidet/pretrained_models/MyModel/MTL_bitrates/"
export EVAL_VAL_FILE="data/svs_val_local.txt"
export EVAL_TEST_FILE="data/svs_test_local.txt"
export EVAL_BN_FILE="data/svs_bn_local.txt"
export FT_TRAIN_FILE="data/svs_bn_local.txt"
export FT_EPOCHS="${FT_EPOCHS:-15}"
export FT_LR="${FT_LR:-5e-5}"
CONFIG="data/params.yaml"
OUT_ROOT="SynWoodscape2021/output"
mkdir -p logs

[ -d "$EVAL_WEIGHTS" ] || { echo "Checkpoint folder not found: $EVAL_WEIGHTS (is the SSD mounted?)"; exit 1; }
[ -f eval/evaluate_mic_mtl.py ] || { echo "Run this from the omnidet root (eval/ not found here)."; exit 1; }

latest_run() { ls -td "$OUT_ROOT"/eval_svs_v2_* 2>/dev/null | head -1; }

case "${1:-}" in
  setup)
    src="${2:?usage: bash run_svs_steps.sh setup <unzipped package folder>}"
    cp "$src"/evaluate_mic_mtl_v2.py "$src"/finetune_svs_receiver.py "$src"/stats_tests.py eval/
    cp "$src"/svs_val_local.txt "$src"/svs_test_local.txt "$src"/svs_bn_local.txt data/
    cp "$src"/run_svs_steps.sh ./
    echo "setup done"
    ;;
  ft1)
    FT_LAMBDAS=0.01 python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_rxft_lam0.01.log
    ;;
  ft_rest)
    FT_LAMBDAS=0.08,0.05,0.03 python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_rxft_rest.log
    ;;
  eval_ft)
    EVAL_RX_FT=1 EVAL_SKIP_CODECS=1 python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/svs_eval_rxft.log
    ;;
  ft1_slim)
    FT_LAMBDAS=0.01 FT_FUSION=slim python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_rxft_slim_lam0.01.log
    ;;
  sx1_slim)
    FT_MODE=sender FT_LAMBDAS=0.01 python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_sxslim_lam0.01.log
    ;;
  eval_slim)
    EVAL_RX_FT=1 EVAL_RX_FT_SUFFIX=_RXFT_SLIM EVAL_SX_SLIM=1 EVAL_SKIP_CODECS=1 \
      python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/svs_eval_slim.log
    ;;
  hevc_ft)
    FT_MODE=hevc FT_HEVC_QPS="${FT_HEVC_QPS:-39,42}" python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/hevc_ft.log
    ;;
  eval_hevc_ft)
    EVAL_RX_FT=1 EVAL_RX_FT_SUFFIX="${EVAL_RX_FT_SUFFIX:-_RXFT_SLIM}" EVAL_CODEC_FT=1 EVAL_CODECS=hevc \
      python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/svs_eval_hevc_ft.log
    ;;
  e2e1)
    FT_MODE=e2e FT_LAMBDAS=0.01 python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_e2e_lam0.01.log
    ;;
  eval_e2e)
    EVAL_E2E=1 EVAL_SX_SLIM=1 EVAL_CODEC_FT=1 EVAL_CODECS=hevc \
      python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/svs_eval_e2e.log
    ;;
  e2e_all)
    if [ "${E2E_ALL_FG:-0}" != 1 ]; then
      E2E_ALL_FG=1 nohup bash "$0" e2e_all > logs/e2e_all.log 2>&1 &
      echo "started in the background (pid $!). Follow it with:  tail -f logs/e2e_all.log"
      exit 0
    fi
    e2e_dir="$EVAL_WEIGHTS/res50_LDMIC0.01_FR_2Classes_yolo3_SVS_E2E/models/weights_$FT_EPOCHS"
    if pgrep -f finetune_svs_receiver.py >/dev/null; then
      echo "[$(date +%H:%M)] training already running, waiting for it to finish"
      while pgrep -f finetune_svs_receiver.py >/dev/null; do sleep 60; done
    elif [ -f "$e2e_dir/decoder.pth" ]; then
      echo "[$(date +%H:%M)] already trained: $e2e_dir"
    else
      echo "[$(date +%H:%M)] training (e2e1)"
      bash "$0" e2e1
    fi
    [ -f "$e2e_dir/decoder.pth" ] || { echo "training did not produce $e2e_dir; stopping"; exit 1; }
    for step in eval_e2e stats pack; do
      echo "[$(date +%H:%M)] $step"
      bash "$0" "$step"
    done
    echo "[$(date +%H:%M)] all done"
    ;;
  final)
    if [ "${FINAL_FG:-0}" != 1 ]; then
      FINAL_FG=1 nohup bash "$0" final > logs/final.log 2>&1 &
      echo "started in the background (pid $!). Follow it with:  tail -f logs/final.log"
      exit 0
    fi
    if ls -d "$EVAL_WEIGHTS"/res50_no_mic_no_compression_FR_2Classes_yolo3_SVS_RXFT_HEVC_QP36/models/weights_* >/dev/null 2>&1; then
      echo "[$(date +%H:%M)] HEVC QP36 already adapted"
    else
      echo "[$(date +%H:%M)] hevc_ft QP36"
      FT_HEVC_QPS=36 bash "$0" hevc_ft
    fi
    for step in eval_e2e stats_vs stats pack; do
      echo "[$(date +%H:%M)] $step"
      bash "$0" "$step"
    done
    echo "[$(date +%H:%M)] all done"
    ;;
  curve)
    if [ "${CURVE_FG:-0}" != 1 ]; then
      CURVE_FG=1 nohup bash "$0" curve > logs/curve.log 2>&1 &
      echo "started in the background (pid $!). Follow it with:  tail -f logs/curve.log"
      exit 0
    fi
    lams="${CURVE_LAMBDAS:-0.08,0.05,0.03}"
    todo() {   # lambdas of $lams whose <run><suffix> folder has no weights yet
      local out="" l
      for l in ${lams//,/ }; do
        ls -d "$EVAL_WEIGHTS"/res50_LDMIC${l}_FR_2Classes_yolo3_SVS$1/models/weights_* >/dev/null 2>&1 || out="$out,$l"
      done
      echo "${out#,}"
    }
    l=$(todo _RXFT_SLIM); if [ -n "$l" ]; then echo "[$(date +%H:%M)] slim receiver: $l"
      FT_LAMBDAS="$l" FT_FUSION=slim python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_rxft_slim_curve.log; fi
    l=$(todo _SXSLIM); if [ -n "$l" ]; then echo "[$(date +%H:%M)] slim sender: $l"
      FT_MODE=sender FT_LAMBDAS="$l" python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_sxslim_curve.log; fi
    l=$(todo _E2E); if [ -n "$l" ]; then echo "[$(date +%H:%M)] task-aware codec: $l"
      FT_MODE=e2e FT_LAMBDAS="$l" python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_e2e_curve.log; fi
    if ! ls -d "$EVAL_WEIGHTS"/res50_no_mic_no_compression_FR_2Classes_yolo3_SVS_RXFT_HEVC_QP33/models/weights_* >/dev/null 2>&1; then
      echo "[$(date +%H:%M)] hevc_ft QP33"
      FT_HEVC_QPS=33 bash "$0" hevc_ft
    fi
    for step in eval_e2e stats_vs stats pack; do
      echo "[$(date +%H:%M)] $step"
      bash "$0" "$step"
    done
    echo "[$(date +%H:%M)] all done"
    ;;
  geo)
    if [ "${GEO_FG:-0}" != 1 ]; then
      GEO_FG=1 nohup bash "$0" geo > logs/geo.log 2>&1 &
      echo "started in the background (pid $!). Follow it with:  tail -f logs/geo.log"
      exit 0
    fi
    while pgrep -f "run_svs_steps.sh curve" >/dev/null || pgrep -f "finetune_svs_receiver.py|evaluate_mic_mtl_v2.py" >/dev/null; do
      echo "[$(date +%H:%M)] GPU busy (curve or another run), waiting"; sleep 120
    done
    has() { ls -d "$EVAL_WEIGHTS"/res50_LDMIC0.01_FR_2Classes_yolo3_SVS$1/models/weights_* >/dev/null 2>&1; }
    if ! has _E2E_CGEO; then echo "[$(date +%H:%M)] task-aware codec + geometry-aware decoder attention"
      FT_MODE=e2e FT_FUSION=slim FT_CODEC_GEO=1 FT_LAMBDAS=0.01 python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_e2e_cgeo.log; fi
    if ! has _SXGEO; then echo "[$(date +%H:%M)] sender, geometry-aware fusion"
      FT_MODE=sender FT_FUSION=geo FT_LAMBDAS=0.01 python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_sxgeo.log; fi
    if ! has _SXNOFUSION; then echo "[$(date +%H:%M)] sender, no fusion"
      FT_MODE=sender FT_FUSION=none FT_LAMBDAS=0.01 python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_sxnofusion.log; fi
    echo "[$(date +%H:%M)] eval: geo sender + geo codec, with HEVC"
    EVAL_E2E=1 EVAL_E2E_SUFFIX=_E2E_CGEO EVAL_SX_SUFFIX=_SXGEO EVAL_CODEC_FT=1 EVAL_CODECS=hevc \
      python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/svs_eval_geo.log
    run=$(latest_run); cp "$run/results.json" logs/results_geo.json
    bash "$0" stats_vs CODEC_HM_HEVC_QP39_adapted "$run"
    bash "$0" stats "$run"
    echo "[$(date +%H:%M)] eval: sender without fusion"
    EVAL_SX_SUFFIX=_SXNOFUSION EVAL_SKIP_CODECS=1 \
      python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/svs_eval_nofusion.log
    run=$(latest_run); cp "$run/results.json" logs/results_nofusion.json
    bash "$0" stats "$run"
    bash "$0" pack
    echo "[$(date +%H:%M)] all done"
    ;;
  geo_finish)   # the no-fusion evaluation of the geo step (it stopped there on 6 Nov), statistics, pack
    EVAL_SX_SUFFIX=_SXNOFUSION EVAL_SKIP_CODECS=1 \
      python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/svs_eval_nofusion.log
    run=$(latest_run); cp "$run/results.json" logs/results_nofusion.json
    bash "$0" stats "$run"
    bash "$0" pack
    ;;
  yskip)
    if [ "${YS_FG:-0}" != 1 ]; then
      YS_FG=1 nohup bash "$0" yskip > logs/yskip.log 2>&1 &
      echo "started in the background (pid $!). Follow it with:  tail -f logs/yskip.log"
      exit 0
    fi
    while pgrep -f "run_svs_steps.sh (curve|geo|highrate)" >/dev/null || pgrep -f "finetune_svs_receiver.py|evaluate_mic_mtl_v2.py" >/dev/null; do
      echo "[$(date +%H:%M)] GPU busy (another run), waiting"; sleep 120
    done
    has() { ls -d "$EVAL_WEIGHTS"/res50_LDMIC0.01_FR_2Classes_yolo3_SVS$1/models/weights_* >/dev/null 2>&1; }
    if ! has _RXFT_SLIM_YSKIP; then echo "[$(date +%H:%M)] receiver with latent skip (codec frozen)"
      FT_FUSION=slim FT_LATENT_SKIP=1 FT_LAMBDAS=0.01 python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_rxft_yskip.log; fi
    if ! has _E2E_YSKIP; then echo "[$(date +%H:%M)] task-aware codec with latent skip"
      FT_MODE=e2e FT_FUSION=slim FT_LATENT_SKIP=1 FT_LAMBDAS=0.01 python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_e2e_yskip.log; fi
    echo "[$(date +%H:%M)] eval: receiver with latent skip, codec frozen"
    EVAL_RX_FT=1 EVAL_RX_FT_SUFFIX=_RXFT_SLIM_YSKIP EVAL_SKIP_CODECS=1 \
      python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/svs_eval_yskip_rx.log
    run=$(latest_run); cp "$run/results.json" logs/results_yskip_rx.json
    bash "$0" stats "$run"
    echo "[$(date +%H:%M)] eval: task-aware codec with latent skip, with HEVC"
    EVAL_E2E=1 EVAL_E2E_SUFFIX=_E2E_YSKIP EVAL_SX_SLIM=1 EVAL_CODEC_FT=1 EVAL_CODECS=hevc \
      python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/svs_eval_yskip_e2e.log
    run=$(latest_run); cp "$run/results.json" logs/results_yskip_e2e.json
    bash "$0" stats_vs CODEC_HM_HEVC_QP39_adapted "$run"
    for sp in val test; do mv "logs/stats_vs_CODEC_HM_HEVC_QP39_adapted_$sp.log" "logs/stats_vs_QP39a_yskip_$sp.log"; done
    bash "$0" stats "$run"
    bash "$0" pack
    echo "[$(date +%H:%M)] all done"
    ;;
  yskip_hr)
    if [ "${YH_FG:-0}" != 1 ]; then
      YH_FG=1 nohup bash "$0" yskip_hr > logs/yskip_hr.log 2>&1 &
      echo "started in the background (pid $!). Follow it with:  tail -f logs/yskip_hr.log"
      exit 0
    fi
    while pgrep -f "run_svs_steps.sh (curve|geo|geo_finish|highrate|yskip)$" >/dev/null || pgrep -f "finetune_svs_receiver.py|evaluate_mic_mtl_v2.py" >/dev/null; do
      echo "[$(date +%H:%M)] GPU busy (another run), waiting"; sleep 120
    done
    has8() { ls -d "$EVAL_WEIGHTS"/res50_LDMIC0.08_FR_2Classes_yolo3_SVS$1/models/weights_* >/dev/null 2>&1; }
    hasq() { ls -d "$EVAL_WEIGHTS"/res50_no_mic_no_compression_FR_2Classes_yolo3_SVS_RXFT_HEVC_QP$1/models/weights_* >/dev/null 2>&1; }
    if ! has8 _RXFT_SLIM_YSKIP; then echo "[$(date +%H:%M)] lambda 0.08 receiver with latent skip"
      FT_FUSION=slim FT_LATENT_SKIP=1 FT_LAMBDAS=0.08 python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_rxft_yskip_lam0.08.log; fi
    qps=""; for q in 31 33; do hasq $q || qps="$qps,$q"; done
    if [ -n "$qps" ]; then echo "[$(date +%H:%M)] hevc_ft ${qps#,}"; FT_HEVC_QPS="${qps#,}" bash "$0" hevc_ft; fi
    for L in native 0.3 1.0; do
      if [ "$L" = native ]; then sfx=_E2E_YSKIP; lam=""; else sfx=_E2E_YSKIP_L$L; lam=$L; fi
      if ! has8 $sfx; then echo "[$(date +%H:%M)] task-aware codec with latent skip, lambda 0.08 -> ${lam:-0.08}"
        FT_MODE=e2e FT_FUSION=slim FT_LATENT_SKIP=1 FT_LAMBDAS=0.08 FT_E2E_LAMBDA=$lam \
          python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_e2e_yskip_lam0.08_$L.log; fi
    done
    for L in native 0.3 1.0; do
      if [ "$L" = native ]; then sfx=_E2E_YSKIP; else sfx=_E2E_YSKIP_L$L; fi
      echo "[$(date +%H:%M)] eval $sfx (lambda 0.08 model)"
      if [ "$L" = native ]; then codec="EVAL_CODEC_FT=1 EVAL_CODECS=hevc"; else codec="EVAL_SKIP_CODECS=1"; fi
      env EVAL_E2E=1 EVAL_E2E_SUFFIX=$sfx $codec \
        python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/svs_eval_yskip_hr_$L.log
      run=$(latest_run); cp "$run/results.json" logs/results_yskip_hr_$L.json
      if [ "$L" = native ]; then
        bash "$0" stats_vs CODEC_HM_HEVC_QP33_adapted "$run"
        for sp in val test; do mv "logs/stats_vs_CODEC_HM_HEVC_QP33_adapted_$sp.log" "logs/stats_vs_QP33a_yskip_hr_$sp.log"; done
      fi
      bash "$0" stats "$run"
    done
    bash "$0" pack
    echo "[$(date +%H:%M)] all done"
    ;;
  highrate)
    if [ "${HR_FG:-0}" != 1 ]; then
      HR_FG=1 nohup bash "$0" highrate > logs/highrate.log 2>&1 &
      echo "started in the background (pid $!). Follow it with:  tail -f logs/highrate.log"
      exit 0
    fi
    while pgrep -f "run_svs_steps.sh (curve|geo)" >/dev/null || pgrep -f "finetune_svs_receiver.py|evaluate_mic_mtl_v2.py" >/dev/null; do
      echo "[$(date +%H:%M)] GPU busy (another run), waiting"; sleep 120
    done
    has8() { ls -d "$EVAL_WEIGHTS"/res50_LDMIC0.08_FR_2Classes_yolo3_SVS$1/models/weights_* >/dev/null 2>&1; }
    hasq() { ls -d "$EVAL_WEIGHTS"/res50_no_mic_no_compression_FR_2Classes_yolo3_SVS_RXFT_HEVC_QP$1/models/weights_* >/dev/null 2>&1; }
    if ! has8 _RXFT_SLIM; then echo "[$(date +%H:%M)] slim receiver for lambda 0.08 (warm start)"
      FT_LAMBDAS=0.08 FT_FUSION=slim python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_rxft_slim_lam0.08.log; fi
    qps=""; for q in 31 33; do hasq $q || qps="$qps,$q"; done
    if [ -n "$qps" ]; then echo "[$(date +%H:%M)] hevc_ft ${qps#,}"; FT_HEVC_QPS="${qps#,}" bash "$0" hevc_ft; fi
    for L in 0.3 1.0; do
      if ! has8 _E2E_L$L; then echo "[$(date +%H:%M)] task-aware codec, lambda 0.08 -> $L"
        FT_MODE=e2e FT_FUSION=slim FT_LAMBDAS=0.08 FT_E2E_LAMBDA=$L python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/svs_e2e_L$L.log; fi
    done
    for L in 0.3 1.0; do
      echo "[$(date +%H:%M)] eval lambda $L"
      if [ "$L" = 0.3 ]; then codec="EVAL_CODEC_FT=1 EVAL_CODECS=hevc"; else codec="EVAL_SKIP_CODECS=1"; fi
      env EVAL_E2E=1 EVAL_E2E_SUFFIX=_E2E_L$L $codec \
        python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/svs_eval_L$L.log
      run=$(latest_run); cp "$run/results.json" logs/results_L$L.json
      if [ "$L" = 0.3 ]; then bash "$0" stats_vs CODEC_HM_HEVC_QP33_adapted "$run"; fi
      bash "$0" stats "$run"
    done
    bash "$0" pack
    echo "[$(date +%H:%M)] all done"
    ;;
  overnight)
    if [ "${ON_FG:-0}" != 1 ]; then
      ON_FG=1 nohup bash "$0" overnight > logs/overnight.log 2>&1 &
      echo "started in the background (pid $!). Follow it with:  tail -f logs/overnight.log"
      exit 0
    fi
    while [ "${NO_WAIT:-0}" != 1 ] && { pgrep -f "run_svs_steps.sh (curve|geo|geo_finish|highrate|yskip|yskip_hr)$" >/dev/null || pgrep -f "finetune_svs_receiver.py|evaluate_mic_mtl_v2.py" >/dev/null; }; do
      echo "[$(date +%H:%M)] GPU busy (another run), waiting"; sleep 120
    done
    export FT_EPOCHS="${ON_EPOCHS:-40}"
    joint() {   # tag source_lambda target_lambda fusion codec_geo(0/1)
      echo "[$(date +%H:%M)] joint $1 (from lambda $2, lambda $3, fusion $4, decoder geometry $5)"
      FT_MODE=joint FT_JOINT_TAG=$1 FT_LAMBDAS=$2 FT_JOINT_LAMBDA=$3 FT_FUSION=$4 FT_CODEC_GEO=$5 \
        python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee "logs/joint_$1.log" \
        || echo "[$(date +%H:%M)] joint $1 FAILED (see logs/joint_$1.log); continuing"
    }
    echo "[$(date +%H:%M)] joint BASE_FT (perception-only control)"
    FT_MODE=joint FT_JOINT_NOCODEC=1 FT_JOINT_TAG=BASE_FT FT_FUSION=none \
      python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/joint_BASE_FT.log \
      || echo "[$(date +%H:%M)] joint BASE_FT FAILED; continuing"
    runs="BASE_FT"
    for spec in "0.01 0.01" "0.03 0.03" "0.08 0.08" "0.08 0.25" "0.08 0.6" "0.08 2.0"; do
      set -- $spec; joint "SWEEP_L$2" $1 $2 slim 1; runs="$runs,SWEEP_L$2"
    done
    joint ABL_NOFUSION 0.03 0.03 none 1;  runs="$runs,ABL_NOFUSION"
    joint ABL_GEO      0.03 0.03 geo 1;   runs="$runs,ABL_GEO"
    joint ABL_NOCGEO   0.03 0.03 slim 0;  runs="$runs,ABL_NOCGEO"
    joint ABL_HEAVY    0.03 0.03 heavy 1; runs="$runs,ABL_HEAVY"
    echo "[$(date +%H:%M)] evaluation (sender side): $runs"
    EVAL_SENDER_ONLY=1 EVAL_JOINT_RUNS="$runs" EVAL_CODEC_PERC_JOINT=BASE_FT \
    EVAL_HEVC_QPS="${ON_HEVC_QPS:-24,27,30,33,36,39,42,45}" \
      python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/overnight_eval.log \
      || echo "[$(date +%H:%M)] evaluation FAILED (see logs/overnight_eval.log)"
    run=$(latest_run); cp "$run/results.json" logs/results_overnight.json || true
    bash "$0" stats_vs JOINT_BASE_FT "$run" || true
    bash "$0" stats "$run" || true
    bash "$0" pack
    echo "[$(date +%H:%M)] all done"
    ;;
  quick)
    {
      echo "[$(date +%H:%M)] QUICK_SIDED_L0.6: side branch, codec loss stops at F3, lambda 0.6, 15 epochs"
      FT_MODE=joint FT_JOINT_TAG=QUICK_SIDED_L0.6 FT_EPOCHS="${Q_EPOCHS:-15}" FT_JOINT_INIT=SWEEP_L0.03 \
        FT_JOINT_LAMBDA=0.6 FT_SIDE=1 FT_DETACH_CODEC=1 python eval/finetune_svs_receiver.py --config "$CONFIG"
      echo "[$(date +%H:%M)] evaluation (sender side, no codecs)"
      EVAL_SENDER_ONLY=1 EVAL_NO_RELEASED=1 EVAL_SKIP_CODECS=1 \
        EVAL_JOINT_RUNS=BASE_FT,SWEEP_L0.03,SWEEP_L0.6,QUICK_SIDED_L0.6 \
        python eval/evaluate_mic_mtl_v2.py --config "$CONFIG"
      run=$(latest_run); cp "$run/results.json" logs/results_quick.json
      python - "$run/results.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("\nmodel                       split  mAP   mIoU   bpp    PSNR   MS-SSIM")
for sp in ("val", "test"):
    for k, m in r[f"SynWoodScape_{sp}"]["models"].items():
        print(f"{k:27s} {sp:5s} {100*m['mAP']:5.1f} {100*m['mIoU_all']:5.1f}  {m.get('BPP', float('nan')):.3f}  "
              f"{m.get('PSNR', float('nan')):5.2f}  {m.get('MS_SSIM', float('nan')):.4f}")
print("\nHEVC (test) for reference: 0.469 bpp 32.98 dB; 0.678 bpp 34.43 dB; 0.964 bpp 35.99 dB")
PY
    } 2>&1 | tee logs/quick.log
    ;;
  stop)
    me=$$
    for p in $(pgrep -f "run_svs_steps.sh"); do [ "$p" != "$me" ] && [ "$p" != "$PPID" ] && kill "$p" 2>/dev/null; done
    sleep 2
    pkill -f "finetune_svs_receiver.py" 2>/dev/null || true
    pkill -f "evaluate_mic_mtl_v2.py" 2>/dev/null || true
    pkill -f "TAppEncoder|TAppDecoder|opj_compress|opj_decompress" 2>/dev/null || true
    for i in 1 2 3 4 5; do
      left=$(pgrep -f "finetune_svs_receiver.py|evaluate_mic_mtl_v2.py" || true)
      [ -z "$left" ] && break
      sleep 2
      [ "$i" -ge 3 ] && pkill -9 -f "finetune_svs_receiver.py|evaluate_mic_mtl_v2.py" 2>/dev/null || true
    done
    left=$(pgrep -af "run_svs_steps.sh|finetune_svs_receiver.py|evaluate_mic_mtl_v2.py" | grep -v " stop$\| restart " || true)
    if [ -n "$left" ]; then echo "still running:"; echo "$left"; else echo "all steps stopped"; fi
    command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    ;;
  full_low)
    if [ "${FL_FG:-0}" != 1 ]; then
      FL_FG=1 nohup bash "$0" full_low > logs/full_low.log 2>&1 &
      echo "started in the background (pid $!). Follow it with:  tail -f logs/full_low.log"
      exit 0
    fi
    while [ "${NO_WAIT:-0}" != 1 ] && { pgrep -f "run_svs_steps.sh (curve|geo|geo_finish|highrate|yskip|yskip_hr|overnight|boost|quick|full)$" >/dev/null || pgrep -f "finetune_svs_receiver.py|evaluate_mic_mtl_v2.py" >/dev/null; }; do
      echo "[$(date +%H:%M)] GPU busy (another run), waiting"; sleep 120
    done
    E="${FU_EPOCHS:-30}"
    for L in ${FL_LAMBDAS:-0.003 0.001}; do
      echo "[$(date +%H:%M)] joint FULL_L$L (lambda $L, slim, side, codec loss stops at F3, $E epochs)"
      env FT_MODE=joint FT_JOINT_TAG=FULL_L$L FT_EPOCHS=$E FT_LAMBDAS=0.03 FT_JOINT_LAMBDA=$L FT_FUSION=slim \
          FT_CODEC_GEO=0 FT_SIDE=1 FT_DETACH_CODEC=1 \
        python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee "logs/joint_FULL_L$L.log" \
        || echo "[$(date +%H:%M)] joint FULL_L$L FAILED; continuing"
    done
    # evaluate only the new low rates, with BASE_FT (statistics reference) and FULL_L0.01 (link to the full sweep).
    # HEVC / JPEG2000 rows are already in results_full.json (same images); FL_CODECS=1 recomputes them here.
    runs="BASE_FT,FULL_L0.01"
    for L in ${FL_LAMBDAS:-0.003 0.001}; do runs="$runs,FULL_L$L"; done
    echo "[$(date +%H:%M)] evaluation (sender side): $runs"
    if [ "${FL_CODECS:-0}" = 1 ]; then codec_env="EVAL_HEVC_QPS=${FL_HEVC_QPS:-36,39,42,45}"; else codec_env="EVAL_SKIP_CODECS=1"; fi
    env EVAL_SENDER_ONLY=1 EVAL_NO_RELEASED=1 EVAL_JOINT_RUNS="$runs" EVAL_CODEC_PERC_JOINT=BASE_FT $codec_env \
      python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/full_low_eval.log \
      || echo "[$(date +%H:%M)] evaluation FAILED"
    run=$(latest_run); cp "$run/results.json" logs/results_full_low.json || true
    bash "$0" stats_vs JOINT_BASE_FT "$run" || true
    for sp in val test; do mv "logs/stats_vs_JOINT_BASE_FT_$sp.log" "logs/stats_vs_BASE_FT_full_low_$sp.log" 2>/dev/null || true; done
    python - "$run/results.json" <<'PY' || true
import json, sys
r = json.load(open(sys.argv[1]))
print("\nmodel                       split  mAP   mIoU   bpp    PSNR   MS-SSIM")
for sp in ("val", "test"):
    for k, m in r[f"SynWoodScape_{sp}"]["models"].items():
        print(f"{k:27s} {sp:5s} {100*m['mAP']:5.1f} {100*m['mIoU_all']:5.1f}  {m.get('BPP', float('nan')):.3f}  "
              f"{m.get('PSNR', float('nan')):5.2f}  {m.get('MS_SSIM', float('nan')):.4f}")
print("\nJPEG2000 (test): 0.160 bpp 24.91 dB; 0.239 bpp 26.06 dB; 0.479 bpp 28.29 dB")
print("HEVC (test):     0.137 bpp 28.63 dB; 0.207 bpp 29.95 dB; 0.311 bpp 31.37 dB; 0.469 bpp 32.98 dB")
PY
    bash "$0" pack
    echo "[$(date +%H:%M)] all done"
    ;;
  restart)
    step="${2:?usage: bash run_svs_steps.sh restart <step>}"
    bash "$0" stop
    sleep 3
    bash "$0" "$step"
    ;;
  full)
    if [ "${FU_FG:-0}" != 1 ]; then
      FU_FG=1 nohup bash "$0" full > logs/full.log 2>&1 &
      echo "started in the background (pid $!). Follow it with:  tail -f logs/full.log"
      exit 0
    fi
    while [ "${NO_WAIT:-0}" != 1 ] && { pgrep -f "run_svs_steps.sh (curve|geo|geo_finish|highrate|yskip|yskip_hr|overnight|boost|quick)$" >/dev/null || pgrep -f "finetune_svs_receiver.py|evaluate_mic_mtl_v2.py" >/dev/null; }; do
      echo "[$(date +%H:%M)] GPU busy (another run), waiting"; sleep 120
    done
    E="${FU_EPOCHS:-30}"
    full_ft() {   # tag lambda fusion side(0/1) detach(0/1)
      echo "[$(date +%H:%M)] joint $1 (lambda $2, fusion $3, side $4, codec loss stops at F3 $5, $E epochs)"
      env FT_MODE=joint FT_JOINT_TAG=$1 FT_EPOCHS=$E FT_LAMBDAS=0.03 FT_JOINT_LAMBDA=$2 FT_FUSION=$3 \
          FT_CODEC_GEO=0 FT_SIDE=$4 FT_DETACH_CODEC=$5 \
        python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee "logs/joint_$1.log" \
        || echo "[$(date +%H:%M)] joint $1 FAILED (see logs/joint_$1.log); continuing"
    }
    if ! ls -d "$EVAL_WEIGHTS"/JOINT/BASE_FT/models/weights_* >/dev/null 2>&1; then
      echo "[$(date +%H:%M)] joint BASE_FT (perception-only control, 40 epochs)"
      FT_MODE=joint FT_JOINT_NOCODEC=1 FT_JOINT_TAG=BASE_FT FT_FUSION=none FT_EPOCHS=40 \
        python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 | tee logs/joint_BASE_FT.log || true
    fi
    full_ft FULL_L0.6  0.6  slim 1 1
    full_ft FULL_L0.1  0.1  slim 1 1
    full_ft FULL_L0.03 0.03 slim 1 1
    full_ft FULL_A_NOFUSION 0.1 none    1 1
    full_ft FULL_A_ADAPTER  0.1 adapter 1 1
    full_ft FULL_A_NOSIDE   0.1 slim    0 1
    full_ft FULL_A_SHARED   0.1 slim    1 0
    full_ft FULL_L0.3  0.3  slim 1 1
    full_ft FULL_L0.01 0.01 slim 1 1
    full_ft FULL_L1.5  1.5  slim 1 1
    full_ft FULL_A_GEO      0.1 geo     1 1
    full_ft FULL_A_HEAVY    0.1 heavy   1 1
    runs="BASE_FT,FULL_L0.01,FULL_L0.03,FULL_L0.1,FULL_L0.3,FULL_L0.6,FULL_L1.5"
    runs="$runs,FULL_A_NOFUSION,FULL_A_ADAPTER,FULL_A_NOSIDE,FULL_A_SHARED,FULL_A_GEO,FULL_A_HEAVY"
    echo "[$(date +%H:%M)] evaluation (sender side): $runs"
    EVAL_SENDER_ONLY=1 EVAL_JOINT_RUNS="$runs" EVAL_CODEC_PERC_JOINT=BASE_FT \
    EVAL_HEVC_QPS="${FU_HEVC_QPS:-22,24,27,30,33,36,39,42,45}" \
      python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/full_eval.log \
      || echo "[$(date +%H:%M)] evaluation FAILED (see logs/full_eval.log)"
    run=$(latest_run); cp "$run/results.json" logs/results_full.json || true
    bash "$0" stats_vs JOINT_BASE_FT "$run" || true
    for sp in val test; do mv "logs/stats_vs_JOINT_BASE_FT_$sp.log" "logs/stats_vs_BASE_FT_full_$sp.log" 2>/dev/null || true; done
    bash "$0" stats "$run" || true
    for sp in val test; do cp "$run/stats_$sp.csv" "logs/stats_full_$sp.csv" 2>/dev/null || true; done
    bash "$0" pack
    echo "[$(date +%H:%M)] all done"
    ;;
  boost)
    if [ "${BO_FG:-0}" != 1 ]; then
      BO_FG=1 nohup bash "$0" boost > logs/boost.log 2>&1 &
      echo "started in the background (pid $!). Follow it with:  tail -f logs/boost.log"
      exit 0
    fi
    while [ "${NO_WAIT:-0}" != 1 ] && { pgrep -f "run_svs_steps.sh (curve|geo|geo_finish|highrate|yskip|yskip_hr|overnight)$" >/dev/null || pgrep -f "finetune_svs_receiver.py|evaluate_mic_mtl_v2.py" >/dev/null; }; do
      echo "[$(date +%H:%M)] GPU busy (another run), waiting"; sleep 120
    done
    run_ft() {   # tag, then environment assignments
      local tag=$1; shift
      echo "[$(date +%H:%M)] joint $tag ($*)"
      env FT_MODE=joint FT_JOINT_TAG=$tag "$@" python eval/finetune_svs_receiver.py --config "$CONFIG" 2>&1 \
        | tee "logs/joint_$tag.log" || echo "[$(date +%H:%M)] joint $tag FAILED (see logs/joint_$tag.log); continuing"
    }
    run_ft ABL_ADAPTER FT_EPOCHS=40 FT_LAMBDAS=0.03 FT_JOINT_LAMBDA=0.03 FT_FUSION=adapter FT_CODEC_GEO=1
    runs="ABL_ADAPTER,QUICK_SIDED_L0.6"
    E="${BO_EPOCHS:-30}"
    for L in ${BO_LAMBDAS:-0.6 0.1 0.03 0.3 0.01 1.5}; do
      run_ft SIDED_L$L FT_EPOCHS=$E FT_JOINT_INIT=SWEEP_L0.03 FT_JOINT_LAMBDA=$L FT_SIDE=1 FT_DETACH_CODEC=1; runs="$runs,SIDED_L$L"
    done
    all="BASE_FT,SWEEP_L0.01,SWEEP_L0.03,SWEEP_L0.08,SWEEP_L0.25,SWEEP_L0.6,SWEEP_L2.0,ABL_NOFUSION,ABL_GEO,ABL_NOCGEO,ABL_HEAVY,$runs"
    echo "[$(date +%H:%M)] evaluation (sender side): $all"
    EVAL_SENDER_ONLY=1 EVAL_JOINT_RUNS="$all" EVAL_CODEC_PERC_JOINT=BASE_FT \
    EVAL_HEVC_QPS="${ON_HEVC_QPS:-24,27,30,33,36,39,42,45}" \
      python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/boost_eval.log \
      || echo "[$(date +%H:%M)] evaluation FAILED (see logs/boost_eval.log)"
    run=$(latest_run); cp "$run/results.json" logs/results_boost.json || true
    bash "$0" stats_vs JOINT_BASE_FT "$run" || true
    for sp in val test; do mv "logs/stats_vs_JOINT_BASE_FT_$sp.log" "logs/stats_vs_BASE_FT_boost_$sp.log" 2>/dev/null || true; done
    bash "$0" stats "$run" || true
    bash "$0" pack
    echo "[$(date +%H:%M)] all done"
    ;;
  stats_vs)
    ref="${2:-CODEC_HM_HEVC_QP39_adapted}"
    run="${3:-$(latest_run)}"
    [ -n "$run" ] || { echo "no evaluation run found in $OUT_ROOT"; exit 1; }
    echo "statistics for $run against $ref"
    for split in val test; do
      python eval/stats_tests.py "$run/SynWoodScape_$split/per_image" --ref "$ref" \
             --csv "$run/stats_vs_${ref}_$split.csv" 2>&1 | tee "logs/stats_vs_${ref}_$split.log"
    done
    ;;
  eval_full)   # sender + receiver as trained, with codecs (already done on 5 Nov; only if needed again)
    python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/svs_eval_full.log
    ;;
  eval_rxbn)   # receiver with BatchNorm statistics from x_hat (already done on 5 Nov)
    EVAL_RX_BN=xhat EVAL_SKIP_CODECS=1 python eval/evaluate_mic_mtl_v2.py --config "$CONFIG" 2>&1 | tee logs/svs_eval_rxbn.log
    ;;
  stats)
    run="${2:-$(latest_run)}"
    [ -n "$run" ] || { echo "no evaluation run found in $OUT_ROOT"; exit 1; }
    echo "statistics for $run"
    for split in val test; do
      python eval/stats_tests.py "$run/SynWoodScape_$split/per_image" --ref SVS_Baseline_no_compression \
             --csv "$run/stats_$split.csv" 2>&1 | tee "logs/stats_$(basename "$run")_$split.log"
    done
    ;;
  pack)
    run="${2:-$(latest_run)}"
    name="svs_results_$(date +%Y%m%d_%H%M).zip"
    tmp="$(mktemp -d)"
    cp "$run/results.json" "$run/run_meta.json" "$tmp"/
    for f in "$run"/stats_*.csv logs/*.log logs/results_*.json; do if [ -f "$f" ]; then cp "$f" "$tmp"/; fi; done
    for m in "$EVAL_WEIGHTS"/*_RXFT*/models/weights_*/rxft_meta.json "$EVAL_WEIGHTS"/*_SXSLIM/models/weights_*/rxft_meta.json "$EVAL_WEIGHTS"/*_E2E*/models/weights_*/rxft_meta.json "$EVAL_WEIGHTS"/*_SXGEO/models/weights_*/rxft_meta.json "$EVAL_WEIGHTS"/*_SXNOFUSION/models/weights_*/rxft_meta.json; do
      if [ -f "$m" ]; then cp "$m" "$tmp/rxft_meta_$(basename "$(dirname "$(dirname "$(dirname "$m")")")").json"; fi
    done
    for m in "$EVAL_WEIGHTS"/JOINT/*/models/weights_*/joint_meta.json; do
      if [ -f "$m" ]; then cp "$m" "$tmp/joint_meta_$(basename "$(dirname "$(dirname "$(dirname "$m")")")").json"; fi
    done
    (cd "$tmp" && zip -q "$OLDPWD/$name" ./*)
    rm -rf "$tmp"
    echo "send me: $(pwd)/$name"
    ;;
  *)
    sed -n '2,106p' "$0"
    ;;
esac
