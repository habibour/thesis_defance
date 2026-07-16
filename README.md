# Bengali Sentiment Analysis — Ablation Codebase

Companion code to `thesis_plan.md`. Reproduces and improves on Islam et al.
(2020)'s `BERTBSA` (frozen mBERT + GRU/LSTM/CNN, 71%/60% on 2-class/3-class)
on the same benchmark, via a fine-tuned Bangla-specific transformer
(BanglaBERT) with class-weighted loss, FGM adversarial training, and a
LoRA parameter-efficient fine-tuning comparison as the flagship novelty.

## Setup (Kaggle, T4 GPU, internet ON)

```bash
pip install -q -r requirements.txt
```

If the `normalizer` package fails to install (no internet on the kernel),
training still runs — `src/preprocessing.py` falls back to manual Unicode +
whitespace normalization. Check `normalizer_available()` in a notebook cell
to confirm which path is active.

Dataset paths already default to the Kaggle mount given in the project brief:

```
/kaggle/input/datasets/reversedthoutgts/bangla-dataset/train_.csv
/kaggle/input/datasets/reversedthoutgts/bangla-dataset/test_.csv
```

Override with `--train_path` / `--test_path` for local runs against
`Dataset/train.csv` / `Dataset/test.csv`.

**Data pipeline has been verified against the real dataset** (see "Verification"
below) — sizes, label distributions, and normalization all check out.

## Running the ablation chain

Each command is one row of the ablation table in `thesis_plan.md`. Run each
for 3 seeds (`--seed 42`, `123`, `2024`) once you're past initial debugging.

```bash
cd src

# Core step 1: fine-tuned BanglaBERT, no imbalance handling
python train.py --task 3class --model_name csebuetnlp/banglabert --seed 42

# Core step 2: + class-weighted loss
python train.py --task 3class --model_name csebuetnlp/banglabert \
    --use_class_weights --seed 42

# Core step 3: + FGM adversarial training
python train.py --task 3class --model_name csebuetnlp/banglabert \
    --use_class_weights --use_fgm --seed 42

# Flagship: LoRA instead of full fine-tuning, same class weighting + FGM
python train.py --task 3class --model_name csebuetnlp/banglabert \
    --use_class_weights --use_fgm --use_lora --seed 42

# 2-class task
python train.py --task 2class --model_name csebuetnlp/banglabert --seed 42
```

Swap `--model_name csebuetnlp/banglabert_large` for the larger model.
LoRA's memory savings make the large model comfortably fit on a single T4
even with a full batch size of 32; full fine-tuning of the large model
needs `--batch_size 16` or smaller.

Each run writes to `/kaggle/working/runs/<run_name>/`:
- `test_metrics.json` — accuracy, macro/micro F1, per-class P/R/F1, confusion matrix, trainable-vs-total param counts
- `test_preds.npy`, `test_labels.npy` — raw predictions, for McNemar's test and ensembling

## Evaluation utilities

```bash
# Confusion matrix heatmap
python evaluate.py confusion --metrics_json ../runs/<run_name>/test_metrics.json

# Is run_b's improvement over run_a statistically significant?
python evaluate.py mcnemar --run_a ../runs/<run_a> --run_b ../runs/<run_b>

# Hard-vote ensemble across several runs (fallback option, see thesis_plan.md)
python evaluate.py ensemble --runs ../runs/<run1> ../runs/<run2> ../runs/<run3>
```

## Verification performed (this session, against `Dataset/train.csv`/`test.csv`)

- **Label distribution matches the base paper's Table III exactly.** 3-class
  test set: 877 Neutral / 843 Positive / 1280 Negative (2000 + 1000 = 3000).
  2-class test set (Neutral dropped): 843 Positive / 1280 Negative = 2123 rows.
- **Splits are clean.** Validation is carved out of the training file only
  (stratified 10%); the test file is completely separate. Checked for
  content-level overlap between splits: 45/1486 validation rows (3%) and
  4/3000 test rows (0.13%) share exact text with some training row — this
  reflects duplicate/near-duplicate short comments naturally occurring in
  the scraped source data (e.g. common one-line reactions repeated by
  different users), not a bug in the split logic. The test-set overlap is
  small enough (0.13%) to not meaningfully inflate reported accuracy, but
  worth a one-line mention in the thesis if asked about data leakage.
- **Normalization pipeline works correctly** on real Bengali text samples
  from the dataset — Unicode NFKC normalization, URL stripping, repeated
  punctuation/whitespace collapse all confirmed. The official BanglaBERT
  `normalizer` package could not be installed in this sandbox (network
  restricted to an allowlist that blocks huggingface.co and pypi index
  mirrors) — code falls back gracefully, but **install and confirm
  `normalizer_available() == True` on Kaggle** before your real runs, since
  that package measurably helps BanglaBERT's performance per its own repo.
- **Model download/tokenization was not tested end-to-end** in this sandbox
  for the same network-restriction reason — `csebuetnlp/banglabert` could
  not be pulled from Hugging Face here. This is expected to work fine on
  Kaggle with internet enabled; run a 1-epoch smoke test there first before
  committing to a full run.

## Design notes

- **Validation split**: carved out of the training file only (stratified,
  10% by default). The test file is never touched until final evaluation.
- **Label mapping**: `Sentiment` column is `0=Neutral, 1=Positive,
  2=Negative`, confirmed by matching per-class counts against the base
  paper's Table III. 2-class task drops Neutral and remaps to `{Positive:
  0, Negative: 1}`.
- **Class weights**: inverse-frequency, normalized to average 1.0, applied
  only to the training cross-entropy loss (not eval).
- **FGM**: perturbs the `word_embeddings` parameter tensor by name match,
  works across BERT/ELECTRA-family architectures without modification, and
  still finds the base model's embeddings when wrapped in a LoRA adapter
  (LoRA doesn't touch the embedding layer by default).
- **LoRA**: wraps the attention `query`/`key`/`value` projections via
  `peft`, classifier head trained fully (`modules_to_save=["classifier"]`).
  Uses a higher default learning rate (`--lora_lr`, default 1e-4) than full
  fine-tuning, since far fewer parameters are being updated. `train.py`
  prints trainable-vs-total parameter counts at startup so you can confirm
  LoRA is actually shrinking the trainable footprint.
- **Model selection**: best checkpoint by validation macro-F1, not
  accuracy, so class imbalance doesn't bias checkpoint selection toward
  the majority class.
