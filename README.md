# StressMAE

Code for the paper **"StressMAE: A Self-Supervised Masked Autoencoder Framework for Stress Detection from Chest-Worn Physiological Signals"** (Milad Ghalejughi, Muhammad Nawaz, Zawar Hussain).

StressMAE is a masked autoencoder for the six signals of the chest-worn RespiBAN device in the WESAD dataset
(ECG, EDA, EMG, respiration, temperature, acceleration). It is trained in two stages and applied to a new person
with a label-free inference step:

1. **Stage 1, self-supervised pre-training.** Masked patch reconstruction (6 of 10 patches of 0.5 s) plus
   cross-modal masking (for half of the windows one sensor is hidden and must be rebuilt from the others).
   Five encoders are trained, each on the complete unlabelled recordings of the 12 subjects outside one group of three.
2. **Stage 2, supervised fine-tuning** on labelled 5 s windows of other subjects (baseline, stress, amusement).
3. **Inference for a new person, without any of its labels:** per-person score calibration, a consensus vote over
   k-means clusters of the person's own frozen embeddings, and Viterbi temporal decoding. The settings of this step
   are chosen on two validation subjects.

The main evaluation is strict leave-one-subject-out (LOSO): the test person is never used for pre-training,
fine-tuning or any model choice.

## Data

WESAD is not included (its licence does not allow redistribution). Download it from the dataset authors
(Schmidt et al., ICMI 2018) and set `WESAD_DIR` to the folder that contains `S2/S2.pkl`, ..., `S17/S17.pkl`.

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
```

The experiments of the paper were run on a CPU (8 cores) with Python 3.13, PyTorch 2.14 and scikit-learn 1.9.

## Reproducing the paper

Run the steps in this order from the repository folder (outputs go to `cache/` and `results/`):

```bash
export WESAD_DIR=/path/to/WESAD

# pre-processing (no labels are used for any fitted step)
python run_experiments.py cache        # single-lead ECG windows (used by the ECG-only ablation)
python make_hrv_cache.py               # handcrafted ECG features of every window (feature baselines)
python cache_multi.py                  # six chest signals at 100 Hz

# Stage 1: pre-training (5 encoders each)
python run_multi.py pretrain           # six-channel StressMAE
python run_experiments.py pretrain     # ECG-only StressMAE (ablation)

# Stage 2: strict LOSO fine-tuning
python run_multi.py loso               # StressMAE (2 seeds) and the same network without pre-training (seed 0)
python run_scratch_seed.py             # second seed of the network without pre-training
python run_ens.py                      # ECG-only StressMAE (3 seeds)
python run_multi.py lowlabel           # label efficiency (1, 5, 10, 25 % of the labels)

# same-person protocols (comparison with the literature only)
python run_multi.py cv --classes 3 --tag v2c3        # random 10-fold, 3 classes
python run_multi.py cv --classes 4 --tag v2c4        # random 10-fold, 4 classes (with meditation)
python run_multi.py cv --classes 3 --tag v2chrono3   # chronological: first 2/3 of each condition trains
python nonstrict_baselines.py

# strict evaluation of all models with the same label-free inference step, statistics, tables
python baselines_fair.py
python make_numbers_v2.py              # writes the tables and numbers used in the manuscript
```

`make_numbers_v2.py` writes into `PAPER_DIR` (default `../stressmae_extracted`); set `PAPER_DIR` to choose another folder.

## Files

| File | Content |
|---|---|
| `run_experiments.py` | ECG pre-processing, ECG-only StressMAE, common training utilities |
| `cache_multi.py` | filtering, down-sampling and per-subject standardization of the six chest signals |
| `run_multi.py` | six-channel StressMAE: pre-training, LOSO fine-tuning, label efficiency, same-person protocols |
| `postproc.py` | score calibration, moving average, Viterbi decoding |
| `select_eval.py` | validation-based choice of the inference settings, consensus vote |
| `baselines_fair.py` | strict comparison of all models with the identical pipeline, Wilcoxon tests with Holm correction |
| `nonstrict_baselines.py` | handcrafted-feature baselines in the same-person protocols |
| `run_ens.py`, `run_scratch_seed.py`, `make_hrv_cache.py` | ablations and baselines |
| `make_numbers_v2.py` | tables, figure and numbers of the manuscript |
| `results/` | per-subject result summaries (JSON) |

## Saved results

`results/` contains the per-subject summaries from which the tables of the manuscript are computed:

| Path | Content |
|---|---|
| `final_strict.json`, `final_strict_stats.json` | strict LOSO scores of all models and the Wilcoxon / Holm tests (written by `baselines_fair.py`) |
| `multi_lowlabel/` | label efficiency (written by `run_multi.py lowlabel`) |
| `multi_cv3_v2/`, `multi_cv4_v2/`, `multi_cv3_chrono_v2/`, `final_nonstrict_baselines.json` | same-person protocols |
| `encoders_multi/multi_group*.json`, `log_multi_pretrain*.txt` | pre-training loss curves and window counts |

Model weights, cached signals and per-window predictions (`*.pt`, `*.npz`) are not included; they are produced by the
commands above. `make_numbers_v2.py` additionally reads the cached signals (`cache/multi`) to report the recording
lengths and class distribution, so it needs the pre-processing step.
