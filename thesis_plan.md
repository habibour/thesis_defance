# Bengali Sentiment Analysis — Final Plan

Defense: 2026-07-29. Target submission: ICCIT 2026 (full paper deadline **2026-07-31**, IEEE Xplore, Cox's Bazar, Dec 18-20 2026 — same conference series as the base paper).

## 1. Accuracy targets

| Task | Target | Basis |
|---|---|---|
| 3-class (Neutral/Positive/Negative) | **75% accuracy** (macro-F1 ~0.70-0.73) | Knowdee's BLP-2023 system hit F1-micro 0.7267 (2nd/30 teams) on a *noisier*, multi-platform social-media 3-class dataset (MUBASE+SentNob). Your dataset is single-source, expert-validated news comments — cleaner data, so matching or beating that on an easier dataset is a realistic, not optimistic, target. |
| 2-class (Positive/Negative) | **85-90% accuracy** | Fine-tuned Bangla-specific transformers on binary sentiment routinely clear 88-95% across multiple 2024-2025 studies on comparable single-domain Bangla datasets (Hoque et al. 2024: 95.97%; Ahmed et al. 2023, Daraz e-commerce: 94.5%; Hamim et al. 2025: 94.21%). 85-90% is a conservative target given your data is *harder* than those (informal news-comment register) but still much cleaner than multi-platform social media. |

Baseline being beaten: base paper (Islam et al., ICCIT 2020) — 71% (2-class), 60% (3-class), frozen mBERT + GRU/LSTM/CNN head.

**Naming caution:** there is a *different* dataset in the literature with an almost identical name — Kowsher et al.'s "Sentiment Analysis on Bengali News Comments" (5-class: slightly-positive/positive/neutral/negative/slightly-negative, ~13,800 rows, their Bangla-BERT gets 84.17% on it). This is **not** your dataset. Your dataset is Islam et al.'s Prothom Alo corpus (3-class, 17,852 rows). Every related-work citation needs to be checked against which of these two it actually used — a mix-up here is an easy, embarrassing catch for a committee member.

## 2. Novelty plan

**Core (non-negotiable, already scoped, low risk):**
1. Fine-tuned BanglaBERT (`csebuetnlp/banglabert`) replacing frozen mBERT — biggest single lever.
2. Class-weighted loss — addresses the imbalance the base paper ignores (train set: Negative 7071 / Positive 3926 / Neutral 3855).
3. FGM adversarial training — borrowed from Knowdee, cheap, measurable.

**Flagship addition — parameter-efficient fine-tuning (LoRA):**
Full fine-tuning a ~110-335M parameter transformer on ~13-15K training rows risks overfitting; a 2025 comparative study on exactly this problem (LoRA vs. IA3 vs. ReFT for low-resource text classification) found full fine-tuning prone to what they term "catastrophic overfitting" in this regime. A related PEFT sentiment-analysis study (MahSA) found PEFT's *largest* gains land specifically on the neutral class and the smallest domain — which is precisely your unsolved problem (Knowdee's own confusion matrices show Neutral F1 stuck at 0.34-0.41 even for their best system). This adds one row to the ablation table (full fine-tune vs. LoRA fine-tune, same class weighting + FGM otherwise) using the `peft` library — a few lines on top of the existing training script, not new architecture. It also reframes "free Kaggle GPU only" from an apology into a real motivation: efficient adaptation of large Bangla LMs under real compute constraints.

**Low-cost polish — explainability layer:**
Attention-weight or SHAP visualization over misclassified examples, added post-hoc to the already-trained final model (no retraining). Matches an explicit 2025 trend in this exact subfield (BanglaSentNet builds its whole framework around this) and slots directly into the error-analysis step already on the schedule.

**Fallback, not flagship — multi-transformer ensemble:**
Soft-voting BanglaBERT-base + BanglaBERT-large (+ optionally XLM-R) has the highest ceiling in the literature (Hoque et al. 2024 hit 95.97% ensembling five transformers on a cleaner dataset), but costs the most compute and time. Reach for this only if the 3-class target isn't hit by day 8 — don't build the plan around it from the start.

**Stretch, only if everything above finishes early:**
Pseudo-labeling using the ~22,500 comments the base paper scraped but discarded during filtering (40,354 scraped → 17,852 kept) — genuinely new since neither the base paper nor Knowdee does this on this dataset.

## 3. Expected training time (Kaggle, single T4, fp16, BanglaBERT-base unless noted)

Rough estimates for ~6 epochs with early stopping on val macro-F1; actual numbers will vary with Kaggle's shared-hardware variance — treat as planning estimates, not guarantees.

| Run | Task | Notes | Approx. time |
|---|---|---|---|
| Fine-tuned baseline | 3-class | ~13.4K train rows | 25-30 min |
| + class-weighted loss | 3-class | same cost as above | 25-30 min |
| + FGM | 3-class | 2x forward/backward per step | 50-60 min |
| LoRA fine-tune | 3-class | fewer trainable params, similar wall-clock | 20-25 min |
| LoRA fine-tune | 3-class, BanglaBERT-**large** | memory headroom from LoRA makes this feasible on one T4 | 45-60 min |
| Full fine-tune | 3-class, BanglaBERT-large | needs batch 16 + grad accumulation | 70-90 min |
| Fine-tuned baseline | 2-class | ~40% fewer rows than 3-class | 15-20 min |
| + FGM | 2-class | | 30-35 min |
| Final config, x3 seeds | either task | multiply the relevant row by 3 | — |
| Explainability pass | either | inference-only, no training | 10-15 min |
| Ensemble evaluation | either | reuses existing checkpoints, just inference + voting | ~5 min |

**Total estimated GPU budget for the full plan (exploratory single-seed runs + 3-seed final runs on the 1-2 winning configs + one large-model trial): roughly 9-12 GPU-hours.** Kaggle's free-tier weekly GPU quota has historically been around 30 hours/week, which comfortably covers this — but confirm the current quota on Kaggle before committing to the schedule below, since platform policies change. Using both GPUs on a T4x2 notebook only helps if you explicitly set up multi-GPU training (HF Trainer will use both via DataParallel automatically if launched as a plain script with two visible GPUs, giving a rough 1.3-1.5x speedup, not a full 2x) — not required to hit these estimates, just a buffer if you fall behind.

## 4. Schedule

Paper writing has to run **in parallel** with experiments, not after — the ICCIT deadline (Jul 31) is two days after your defense (Jul 29), so there's no separate "write it up afterward" phase like a normal thesis timeline would allow.

**Day 1-2 (Jul 16-17): Setup**
- Kaggle notebook, T4 GPU, `transformers`/`datasets`/`peft`/BanglaBERT normalizer installed.
- `src/` pipeline: preprocessing, dataset loading (2-class + 3-class variants), training script, evaluation script.
- Push skeleton to `github.com/habibour/thesis_defance`.
- Start the paper skeleton: title, abstract placeholder, intro, related work (you already have the citations from this conversation — Islam et al. 2020, Kowsher et al. 2022, Knowdee/BLP-2023, Hoque et al. 2024, BanglaSentNet 2025, the LoRA/PEFT studies).

**Day 3-4 (Jul 18-19): Core ablation, step 1**
- Fine-tuned BanglaBERT baseline, both tasks. Log accuracy/macro-F1/confusion matrix.
- Fill in the Methodology section draft while this runs.

**Day 5-6 (Jul 20-21): Core ablation, steps 2-3**
- Add class-weighted loss, then FGM. Re-run both tasks.
- Update the paper's results table skeleton as each number comes in — don't leave this for later.

**Day 7 (Jul 22): Flagship novelty — LoRA**
- LoRA fine-tune, both tasks, same class weighting + FGM setup for a clean comparison row.
- If 3-class is already at/near target, also try LoRA on BanglaBERT-large here since the memory headroom makes it cheap.

**Day 8 (Jul 23): Explainability + go/no-go on ensemble**
- Attention/SHAP pass on misclassified examples for the error-analysis chapter.
- Check 3-class number against the 75% target. If short, this is when you reach for the ensemble fallback — not before.

**Day 9 (Jul 24): Final runs**
- 3 seeds on the 1-2 winning configs per task, for mean ± std.
- McNemar's test: best model vs. reproduced frozen-mBERT baseline, same test set.

**Day 10 (Jul 25): Error analysis + Results/Discussion draft**
- Confusion matrices, per-class F1, misclassified examples (expect Neutral confusion to be the hardest problem, consistent with every paper surveyed).
- Write Results and Discussion sections in full — not a draft, the actual submission text.

**Day 11 (Jul 26): Slides + paper tightening**
- Defense slides: problem → base paper's gap → fixes → ablation results → error analysis → conclusion.
- Full read-through of the paper draft against IEEE conference formatting.

**Day 12 (Jul 27): Rehearsal + repo/paper polish**
- Mock defense; anticipate "isn't this just a bigger model" and have the diagnostic-contribution framing ready.
- Repo cleanup (README, requirements, results/ folder).
- Paper: abstract finalized, references formatted, co-author/supervisor sign-off requested.

**Day 13 (Jul 28): Buffer**
- Absorb slippage. Final proofread of the paper.

**Jul 29: Defense.**
**Jul 30: Fold in any defense feedback that's fast to incorporate.**
**Jul 31: Submit to ICCIT 2026.**

If July 31 turns out unrealistic once you see real progress, ECCT 2026 and ICECTE 2026 (RUET) are other IEEE-Bangladesh-region venues to check for later deadlines — not yet verified, check their sites directly before relying on them.

## 5. Comparison methodology & defensibility

Your `Dataset/train.csv`/`test.csv` are the identical 17,852-row Prothom Alo dataset and 3,000-row test split the base paper used (row counts and per-class counts match their Table III exactly) — report their numbers directly, no re-derivation needed.

**Comparison table:**

| Model | PLM | Fine-tuned? | 2-class Acc | 3-class Acc | 3-class Macro-F1 |
|---|---|---|---|---|---|
| Sharfuddin et al. (tf-idf+BiLSTM) | — | — | 68% | — | — |
| BERTBSA, Islam et al. 2020 (base paper) | mBERT | frozen | 71% | 60% | — |
| Yours: fine-tuned BanglaBERT | BanglaBERT | full | ? | ? | ? |
| + class-weighted loss | | | | | |
| + FGM | | | | | |
| + LoRA (flagship) | | | | | |
| + ensemble (if needed) | | | | | |

**Rigor requirements:**
- Report macro-F1 alongside accuracy, especially 3-class, since accuracy alone can hide majority-class bias on this imbalanced data.
- 3-5 seeds on final configs, mean ± std reported.
- McNemar's test between best model and reproduced frozen-mBERT baseline on the identical test set.

**Anticipated pushback:** "You just swapped in a bigger/better pretrained model." Answer: the contribution is isolating and quantifying *which specific design choice* in a widely-cited 2020 paper caused its accuracy ceiling, on the exact benchmark that paper introduced — a diagnostic contribution. Precedent: Knowdee's BLP-2023 paper (2nd/30 at a peer-reviewed workshop) is also just fine-tuning + class handling + ensembling with no new architecture.

**Framing sentence:** "Islam et al. (2020) established the standard Bengali SA benchmark but froze a multilingual encoder and reported no imbalance handling — leaving unclear how much of their 60/71% ceiling was a data limitation versus a methodology limitation. We answer this with a controlled ablation on their exact benchmark, and additionally show that parameter-efficient adaptation recovers most of full fine-tuning's gains at a fraction of the compute, with the largest improvement on the previously hardest class."

## 6. Contributions (state explicitly in the paper)

1. Ablation study attributing the base paper's accuracy gap to specific causes (frozen vs. fine-tuned encoder, multilingual vs. Bangla-specific pretraining, no imbalance handling).
2. New results on a standing public benchmark, directly comparable to the base paper and the tf-idf+BiLSTM baseline it cites.
3. Transfer study of BLP-2023-winning techniques (FGM, class-weighted fine-tuning) applied to a different (cleaner, single-source) dataset than they were designed for.
4. A parameter-efficient fine-tuning comparison (full vs. LoRA) on this benchmark, evaluated specifically for its effect on the hardest class (Neutral) and its compute-efficiency tradeoff — relevant to any low-resource-language NLP work under real GPU constraints.
5. Error analysis + explainability characterizing remaining failure modes.
6. A reproducible, modernized codebase (base paper's code uses deprecated `torchtext` APIs and never unfreezes BERT).

## 7. Repo structure

```
thesis_defance/
  data/            # small samples + README pointing to Kaggle dataset
  src/
    preprocessing.py
    dataset.py
    train.py         # supports --use_class_weights, --use_fgm, --use_lora flags
    evaluate.py       # confusion matrix, McNemar's test, ensemble voting
  notebooks/       # Kaggle notebooks actually used, one per major experiment
  results/         # metrics.json, confusion matrices, plots per experiment
  paper/           # manuscript draft, IEEE template
  README.md        # reproduction instructions
```

## 8. Risk mitigation

If targets aren't hit by day 8 despite the full technique stack, the ablation study itself — with a large, well-documented jump over the frozen-mBERT baseline and a clean PEFT-vs-full-fine-tune comparison — is still a legitimate, defensible, publishable contribution on its own. Don't let chasing the last 1-2 points eat into days 9-13, which are needed for writing, slides, and rehearsal, or into the Jul 31 submission window.

## 9. Related work assembled this conversation

- Islam et al. 2020, *Sentiment analysis in Bengali via transfer learning using multi-lingual BERT* — base paper.
- Liu et al. 2023 (Knowdee), BLP-2023 Task 2 — FGM, ensembling, pseudo-labeling; F1-micro 0.7267 on MUBASE+SentNob.
- Kowsher et al. 2022, *Bangla-BERT: Transformer-Based Efficient Model* — monolingual Bangla BERT; evaluated on a differently-named but distinct 5-class "Bengali News Comments" dataset (84.17% acc) — cite carefully, don't conflate with your dataset.
- Hoque et al. 2024, *Exploring transformer models in the sentiment analysis task for the under-resource Bengali language* — 5-model transformer ensemble, 95.97% accuracy on their (cleaner, smaller) dataset — supports the ensemble-fallback option.
- Hamim et al. 2025 (ECAI) — Flair + BanglaBERT embeddings, 94.21%, benchmarked against classical ML and GPT-2 — a template for a thorough baseline-comparison table.
- Islam et al. 2025, *BanglaSentNet* — explainable hybrid ensemble with SHAP + attention visualization, cross-domain/few-shot robustness testing — template for the explainability step.
- 2025 Frontiers in Big Data study comparing LoRA/IA3/ReFT for low-resource text classification — motivates the PEFT flagship angle, notes full fine-tuning's overfitting risk in this regime.
- MahSA (2024-2025) — PEFT-based sentiment analysis with largest gains on the neutral class and smallest domain — directly motivates why LoRA specifically targets your hardest class.
