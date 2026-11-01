#!/usr/bin/env bash
# Accuracy runs on the four benchmark graphs (Tables 2, 5, and 6; Figure 4). For every configuration and seed, QTem
# and the full-precision pipeline (ETC) train with the same configuration file and seed, and hence with the same
# initialization, features, sampler, and negatives. Table 2 lists the mean test AP of the three QTem runs of a
# configuration. mrr.sh and so_apan.sh evaluate the checkpoints in models/ again.
# Run from the repository root and select the GPU with CUDA_VISIBLE_DEVICES; DRY=1 prints the commands only.
set -u
L=logs/accuracy
mkdir -p models $L
run () {  # <name> <arguments of train.py>
  local n=$1; shift
  echo "python -u train.py $* --model_name $n"
  [ "${DRY:-0}" = 1 ] || python -u train.py "$@" --model_name "$n" > "$L/$n.log" 2>&1
}
for M in TGN TGAT APAN; do
  for S in 0 1 2; do
    # LastFM: 8-bit codes, one scale per row of 128 values
    run acc_LASTFM_${M}_fp32_s$S --data LASTFM --config config/main/LastFM_$M.yml --seed $S --qbits 0
    run acc_LASTFM_${M}_qtem_s$S --data LASTFM --config config/main/LastFM_$M.yml --seed $S --qbits 8 --qblock 128
    # Wiki-Talk and Stack-Overflow: 8-bit codes, one scale per block of four values
    run acc_WIKITALK_${M}_fp32_s$S --data WIKITALK --config config/main/WikiTalk_$M.yml --seed $S --qbits 0
    run acc_WIKITALK_${M}_qtem_s$S --data WIKITALK --config config/main/WikiTalk_$M.yml --seed $S --qbits 8 --qblock 4
    run acc_STACKOVERFLOW_${M}_fp32_s$S --data STACKOVERFLOW --config config/main/StackOv_$M.yml --seed $S --qbits 0
    run acc_STACKOVERFLOW_${M}_qtem_s$S --data STACKOVERFLOW --config config/main/StackOv_$M.yml --seed $S --qbits 8 --qblock 4
  done
  # GDELT: QTem keeps the binary tables as lossless bit planes; the full-precision pipeline keeps one byte per value and
  # converts the gathered rows to 32-bit floats, so both read identical values
  for S in 0 1 7; do
    run acc_GDELT_${M}_fp32_s$S --data GDELT --config config/main/GDELT_$M.yml --seed $S --boolfeat 1
    run acc_GDELT_${M}_qtem_s$S --data GDELT --config config/main/GDELT_$M.yml --seed $S --qpack 1
  done
done
