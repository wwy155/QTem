#!/usr/bin/env bash
# Checks on Stack-Overflow-APAN (Section 6.2; Appendix D): a second full-precision run of seed 0; the checkpoints of
# seeds 0 and 1 of accuracy.sh evaluated with the inputs of either pipeline (test AP) and with the inputs of the other
# pipeline against 49 negatives per positive (MRR); and the second run evaluated against 49 negatives.
# Needs the checkpoints of accuracy.sh. Run from the repository root and select the GPU with CUDA_VISIBLE_DEVICES;
# DRY=1 prints the commands only.
set -u
L=logs/so_apan
mkdir -p models $L
run () {  # <name> <arguments of train.py>
  local n=$1; shift
  echo "python -u train.py $* --model_name $n"
  [ "${DRY:-0}" = 1 ] || python -u train.py "$@" --model_name "$n" > "$L/$n.log" 2>&1
}
C=config/main/StackOv_APAN.yml
FP32="--qbits 0"
QTEM="--qbits 8 --qblock 4"
run so_apan_fp32_s0_repeat --data STACKOVERFLOW --config $C --seed 0 $FP32
run mrr_so_apan_fp32_s0_repeat --data STACKOVERFLOW --config $C --seed 0 $FP32 \
  --eval_only 1 --ckpt models/so_apan_fp32_s0_repeat.pkl --eval_neg_samples 49
for S in 0 1; do
  for CK in fp32 qtem; do
    for IN in fp32 qtem; do
      if [ $IN = fp32 ]; then X=$FP32; else X=$QTEM; fi
      run swap_ck${CK}_in${IN}_s$S --data STACKOVERFLOW --config $C --seed $S $X \
        --eval_only 1 --ckpt models/acc_STACKOVERFLOW_APAN_${CK}_s$S.pkl
    done
  done
  run swap49_ckfp32_inqtem_s$S --data STACKOVERFLOW --config $C --seed $S $QTEM \
    --eval_only 1 --ckpt models/acc_STACKOVERFLOW_APAN_fp32_s$S.pkl --eval_neg_samples 49
  run swap49_ckqtem_infp32_s$S --data STACKOVERFLOW --config $C --seed $S $FP32 \
    --eval_only 1 --ckpt models/acc_STACKOVERFLOW_APAN_qtem_s$S.pkl --eval_neg_samples 49
done
