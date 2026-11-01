#!/usr/bin/env bash
# Data-path variants of Section 6.5 (Table 13; Figures 1 and 6). In each of four rounds, every variant of a
# configuration trains one epoch (seed 0, without validation) on the same GPU, back to back, in the listed order in odd
# rounds and in reverse order in even rounds. SIMPLE, and on LastFM also TGL, ran in the same rounds with their own
# code (Table 13).
# Run from the repository root and select the GPU with CUDA_VISIBLE_DEVICES; DRY=1 prints the commands only.
set -u
L=logs/datapath
mkdir -p models $L
run () {  # <name> <arguments of train.py>
  local n=$1; shift
  echo "python -u train.py $* --model_name $n"
  [ "${DRY:-0}" = 1 ] || python -u train.py "$@" --model_name "$n" > "$L/$n.log" 2>&1
}
variant () {  # <variant> <QTem codec>: its arguments of train.py
  case $1 in
    etc)    echo "--qbits 0" ;;              # ETC
    etcpin) echo "--qbits 0 --pin 1" ;;      # ETC+pin
    f32g)   echo "--qbits 0 --qgpu 1" ;;     # FP32-G
    h16pin) echo "--half 1 --pin 1" ;;       # FP16+pin
    h16g)   echo "--half 1 --qgpu 1" ;;      # FP16-G
    u8pin)  echo "--boolfeat 1 --pin 1" ;;   # uint8+pin
    u8g)    echo "--boolfeat 1 --qgpu 1" ;;  # uint8-G
    qhpin)  echo "$2 --pin 1" ;;             # QTem-H
    qgpu)   echo "$2 --qgpu 1" ;;            # QTem-G
  esac
}
# configuration|data|configuration file (one epoch)|QTem codec|variants in listed order
for spec in \
  "lastfm_tgn|LASTFM|config/main/LastFM_TGN_1E.yml|--qbits 8 --qblock 128|etc etcpin f32g qhpin qgpu" \
  "lastfm_tgat|LASTFM|config/main/LastFM_TGAT_1E.yml|--qbits 8 --qblock 128|etc etcpin f32g qhpin qgpu" \
  "wiki_tgn|WIKITALK|config/main/WikiTalk_TGN_1E.yml|--qbits 8 --qblock 4|etc etcpin f32g qhpin qgpu" \
  "wiki_tgat|WIKITALK|config/main/WikiTalk_TGAT_1E.yml|--qbits 8 --qblock 4|etc etcpin f32g qhpin qgpu" \
  "so_tgat|STACKOVERFLOW|config/main/StackOv_TGAT_1E.yml|--qbits 8 --qblock 4|etc etcpin h16pin h16g qhpin qgpu" \
  "so_tgn|STACKOVERFLOW|config/main/StackOv_TGN_1E.yml|--qbits 8 --qblock 4|etc etcpin h16g qhpin qgpu" \
  "gdelt_tgat|GDELT|config/main/GDELT_TGAT_1E.yml|--qpack 1|qgpu qhpin u8pin u8g etc etcpin"; do
  IFS='|' read -r T D C Q V <<< "$spec"
  read -r -a A <<< "$V"
  for R in 1 2 3 4; do
    if [ $((R % 2)) = 1 ]; then O=("${A[@]}"); else O=(); for ((i = ${#A[@]} - 1; i >= 0; i--)); do O+=("${A[$i]}"); done; fi
    for v in "${O[@]}"; do
      run dp_${T}_${v}_r$R --data $D --config $C --seed 0 $(variant $v "$Q") --skip_eval 1
    done
  done
done
