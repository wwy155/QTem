#!/usr/bin/env bash
# Evaluation of the checkpoints of accuracy.sh and real.sh against 49 random negatives per positive (MRR).
# Every checkpoint is evaluated with its seed and with the representation it was trained with.
# Run from the repository root and select the GPU with CUDA_VISIBLE_DEVICES; DRY=1 prints the commands only.
set -u
L=logs/mrr
mkdir -p models $L
run () {  # <name> <arguments of train.py>
  local n=$1; shift
  echo "python -u train.py $* --model_name $n"
  [ "${DRY:-0}" = 1 ] || python -u train.py "$@" --model_name "$n" > "$L/$n.log" 2>&1
}
ev () {  # <data> <config> <model> <seed> <fp32|qtem> <arguments of the representation>
  local D=$1 C=$2 M=$3 S=$4 A=$5; shift 5
  run mrr_${D}_${M}_${A}_s$S --data $D --config $C --seed $S "$@" \
    --eval_only 1 --ckpt models/acc_${D}_${M}_${A}_s$S.pkl --eval_neg_samples 49
}
for M in TGN TGAT APAN; do
  for S in 0 1 2; do
    ev LASTFM config/main/LastFM_$M.yml $M $S fp32 --qbits 0
    ev LASTFM config/main/LastFM_$M.yml $M $S qtem --qbits 8 --qblock 128
    ev WIKITALK config/main/WikiTalk_$M.yml $M $S fp32 --qbits 0
    ev WIKITALK config/main/WikiTalk_$M.yml $M $S qtem --qbits 8 --qblock 4
    ev STACKOVERFLOW config/main/StackOv_$M.yml $M $S fp32 --qbits 0
    ev STACKOVERFLOW config/main/StackOv_$M.yml $M $S qtem --qbits 8 --qblock 4
    for D in WIKI REDDIT; do
      ev $D config/real/${D}_$M.yml $M $S fp32 --qbits 0
      ev $D config/real/${D}_$M.yml $M $S qtem --qbits 8 --qblock 172
    done
  done
done
