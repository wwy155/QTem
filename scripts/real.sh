#!/usr/bin/env bash
# Accuracy runs on Wikipedia and Reddit with their real 172-dimensional edge features (Table 7; Figure 4): the
# full-precision pipeline and QTem with 8-bit and with 4-bit codes (one scale per row), with the same configuration
# file and seed. DATA/WIKI and DATA/REDDIT hold TGL's files. mrr.sh evaluates the checkpoints in models/ again.
# Run from the repository root and select the GPU with CUDA_VISIBLE_DEVICES; DRY=1 prints the commands only.
set -u
L=logs/real
mkdir -p models $L
run () {  # <name> <arguments of train.py>
  local n=$1; shift
  echo "python -u train.py $* --model_name $n"
  [ "${DRY:-0}" = 1 ] || python -u train.py "$@" --model_name "$n" > "$L/$n.log" 2>&1
}
for D in WIKI REDDIT; do
  for M in TGN TGAT APAN; do
    for S in 0 1 2; do
      C=config/real/${D}_$M.yml
      run acc_${D}_${M}_fp32_s$S --data $D --config $C --seed $S --qbits 0
      run acc_${D}_${M}_qtem_s$S --data $D --config $C --seed $S --qbits 8 --qblock 172
      run acc_${D}_${M}_qtem4_s$S --data $D --config $C --seed $S --qbits 4 --qblock 172
    done
  done
done
