#!/usr/bin/env bash
# Epoch times of QTem in Table 2. For every configuration, QTem (records in GPU memory), ETC, and QTem again train back
# to back on one GPU with the configuration files in config/timing, without validation, with the number of host
# threads and epochs listed below; on Wiki-Talk with 172-dimensional random edge features. Table 2 lists the mean of
# the epoch times ("[timing] wall train-loop time" in the logs) over both QTem runs.
# Run from the repository root and select the GPU with CUDA_VISIBLE_DEVICES; DRY=1 prints the commands only.
set -u
L=logs/timing
mkdir -p models $L
run () {  # <name> <environment> <arguments of train.py>
  local n=$1 e=$2; shift 2
  echo "$e python -u train.py $* --model_name $n"
  [ "${DRY:-0}" = 1 ] || env $e python -u train.py "$@" --model_name "$n" > "$L/$n.log" 2>&1
}
# configuration|data|model|QTem codec|host threads|epochs of QTem, ETC, QTem
for spec in \
  "lastfm_tgn|LASTFM|TGN|--qbits 8 --qblock 128|32|3 3 3" \
  "lastfm_tgat|LASTFM|TGAT|--qbits 8 --qblock 128|1|3 3 3" \
  "lastfm_apan|LASTFM|APAN|--qbits 8 --qblock 128|1|3 3 3" \
  "wiki_tgn|WIKITALK|TGN|--qbits 8 --qblock 4|4|3 2 3" \
  "wiki_tgat|WIKITALK|TGAT|--qbits 8 --qblock 4|4|3 2 3" \
  "wiki_apan|WIKITALK|APAN|--qbits 8 --qblock 4|4|3 2 3" \
  "so_tgat|STACKOVERFLOW|TGAT|--qbits 8 --qblock 4|4|2 1 1" \
  "gdelt_tgat|GDELT|TGAT|--qpack 1|4|2 1 1" \
  "so_tgn|STACKOVERFLOW|TGN|--qbits 8 --qblock 4|4|2 1 1" \
  "gdelt_tgn|GDELT|TGN|--qpack 1|4|2 1 1" \
  "so_apan|STACKOVERFLOW|APAN|--qbits 8 --qblock 4|4|2 1 1" \
  "gdelt_apan|GDELT|APAN|--qpack 1|4|2 1 1"; do
  IFS='|' read -r T D M Q H E <<< "$spec"
  read -r E1 E2 E3 <<< "$E"
  X="QTEM_CPU_THREADS=$H"
  if [ $D = WIKITALK ]; then X="$X QTEM_WIKITALK_EDGE_FEATS=1"; fi
  C=config/timing/${M}_$D.yml
  run time_${T}_qtem1 "$X QTEM_TIMING_EPOCHS=$E1" --data $D --config $C --seed 0 $Q --qgpu 1 --skip_eval 1
  run time_${T}_etc "$X QTEM_TIMING_EPOCHS=$E2" --data $D --config $C --seed 0 --qbits 0 --skip_eval 1
  run time_${T}_qtem2 "$X QTEM_TIMING_EPOCHS=$E3" --data $D --config $C --seed 0 $Q --qgpu 1 --skip_eval 1
done
