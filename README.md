# QTem

Code for the paper **QTem: Bounded-Error Compressed Data Movement for Temporal Graph Neural Network Training**.
Appendix of the paper: [appendix.pdf](appendix.pdf).

QTem is built on [ETC](https://github.com/eddiegaoo/ETC) and [TGL](https://github.com/amazon-science/tgl)
(TGL's license: [LICENSE-TGL](LICENSE-TGL), [NOTICE-TGL](NOTICE-TGL)).

## Setup

Python 3.10, PyTorch 2.1, DGL 2.4, NumPy, pandas, PyYAML, scikit-learn, pybind11.

    python setup.py build_ext --inplace
    mkdir -p models

Datasets as in ETC: `bash down.sh` downloads LastFM and GDELT; Wiki-Talk and Stack-Overflow come from SNAP and are
converted with `python preprocess.py --data WIKITALK --txt <file>` (or `STACKOVERFLOW`). Wikipedia and Reddit are TGL's
files in `DATA/WIKI` and `DATA/REDDIT`.

## Run

    # QTem, 8-bit codes (LastFM: one scale per row; Wiki-Talk, Stack-Overflow: --qblock 4), tables in GPU memory
    CUDA_VISIBLE_DEVICES=0 python train.py --data LASTFM --config config/main/LastFM_TGN.yml --qbits 8 --qblock 128 --qgpu 1 --seed 0
    CUDA_VISIBLE_DEVICES=0 python train.py --data STACKOVERFLOW --config config/main/StackOv_TGAT.yml --qbits 8 --qblock 4 --qgpu 1 --seed 0

    # QTem, lossless bit planes for GDELT's binary features
    CUDA_VISIBLE_DEVICES=0 python train.py --data GDELT --config config/main/GDELT_TGN.yml --qpack 1 --qgpu 1 --seed 0

    # full precision (FP32)
    CUDA_VISIBLE_DEVICES=0 python train.py --data LASTFM --config config/main/LastFM_TGN.yml --qbits 0 --seed 0

Other options: `python train.py --help`.

## Experiments

The commands of the experiments in the paper are in `scripts/`.
