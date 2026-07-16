"""
Training entrypoint for the Bengali sentiment analysis ablation study.

Each ablation row in thesis_plan.md maps to a flag combination here:

  Core step 1 (fine-tuned BanglaBERT):
    python train.py --task 3class --model_name csebuetnlp/banglabert

  Core step 2 (+ class-weighted loss):
    python train.py --task 3class --model_name csebuetnlp/banglabert --use_class_weights

  Core step 3 (+ FGM):
    python train.py --task 3class --model_name csebuetnlp/banglabert --use_class_weights --use_fgm

  Flagship (LoRA instead of full fine-tune, same class weighting + FGM):
    python train.py --task 3class --model_name csebuetnlp/banglabert \
        --use_class_weights --use_fgm --use_lora

  2-class task:
    python train.py --task 2class --model_name csebuetnlp/banglabert

Run with different --seed values (e.g. 42, 123, 2024) and average results for
the mean +/- std reporting called for in thesis_plan.md.

Default paths point at the Kaggle dataset mount given in the project brief:
  /kaggle/input/datasets/reversedthoutgts/bangla-dataset/train_.csv
  /kaggle/input/datasets/reversedthoutgts/bangla-dataset/test_.csv
Override with --train_path / --test_path for local runs (e.g. against
Dataset/train.csv and Dataset/test.csv).
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import build_datasets, class_weights_from_labels  # noqa: E402
from weighted_fgm_trainer import WeightedFGMTrainer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["3class", "2class"], default="3class")
    p.add_argument("--model_name", default="csebuetnlp/banglabert")
    p.add_argument(
        "--train_path",
        default="/kaggle/input/datasets/reversedthoutgts/bangla-dataset/train_.csv",
    )
    p.add_argument(
        "--test_path",
        default="/kaggle/input/datasets/reversedthoutgts/bangla-dataset/test_.csv",
    )
    p.add_argument("--output_dir", default="/kaggle/working/runs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--epochs", type=float, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--use_fgm", action="store_true")
    p.add_argument("--fgm_epsilon", type=float, default=1.0)

    # LoRA / parameter-efficient fine-tuning
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["query", "key", "value"],
        help="Attention submodule names to wrap with LoRA adapters "
        "(defaults match standard BERT/ELECTRA attention naming, which "
        "BanglaBERT follows since it's an ELECTRA-family model).",
    )
    # LoRA typically tolerates a higher LR than full fine-tuning since far
    # fewer parameters are being updated; override with --lr if you sweep.
    p.add_argument("--lora_lr", type=float, default=1e-4)

    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    return p.parse_args()


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    return {"accuracy": acc, "macro_f1": macro_f1, "micro_f1": micro_f1}


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def maybe_wrap_with_lora(model, args):
    if not args.use_lora:
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        # Classifier head is small; train it fully rather than adapting it,
        # LoRA only needs to cover the frozen backbone's attention layers.
        modules_to_save=["classifier"],
    )
    try:
        model = get_peft_model(model, lora_config)
    except ImportError as e:
        # peft's LoRA dispatcher unconditionally probes torchao and raises
        # instead of falling back when an older torchao is installed (e.g.
        # Kaggle's base image ships 0.10.0). We never use torchao/quantized
        # LoRA here, so stub the check out and retry once.
        if "torchao" not in str(e):
            raise
        import peft.tuners.lora.torchao as _peft_lora_torchao

        _peft_lora_torchao.is_torchao_available = lambda: False
        model = get_peft_model(model, lora_config)
    return model


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    run_name = args.run_name or (
        f"{args.task}_{args.model_name.split('/')[-1]}"
        f"{'_cw' if args.use_class_weights else ''}"
        f"{'_fgm' if args.use_fgm else ''}"
        f"{'_lora' if args.use_lora else ''}_seed{args.seed}"
    )
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[run] {run_name}")
    print(f"[data] train={args.train_path} test={args.test_path}")

    bundle = build_datasets(
        train_path=args.train_path,
        test_path=args.test_path,
        task=args.task,
        val_size=args.val_size,
        seed=args.seed,
    )
    ds, num_labels, label_names = bundle.dataset_dict, bundle.num_labels, bundle.label_names
    print(f"[data] sizes: {[(k, len(v)) for k, v in ds.items()]}, num_labels={num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    model = maybe_wrap_with_lora(model, args)

    trainable, total = count_trainable_params(model)
    print(f"[params] trainable={trainable:,} / total={total:,} ({100*trainable/total:.2f}%)")

    class_weights = None
    if args.use_class_weights:
        class_weights = class_weights_from_labels(tokenized["train"]["label"], num_labels)
        print(f"[class weights] {class_weights.tolist()}")

    lr = args.lora_lr if args.use_lora and args.lr == 2e-5 else args.lr

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
    )

    trainer = WeightedFGMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_fn,
        class_weights=class_weights,
        use_fgm=args.use_fgm,
        fgm_epsilon=args.fgm_epsilon,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    # Final evaluation on the held-out test set (never seen during training/val).
    test_output = trainer.predict(tokenized["test"])
    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_labels = test_output.label_ids

    acc = accuracy_score(test_labels, test_preds)
    macro_f1 = f1_score(test_labels, test_preds, average="macro")
    micro_f1 = f1_score(test_labels, test_preds, average="micro")
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        test_labels, test_preds, average=None, labels=list(range(num_labels))
    )
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(num_labels)))

    metrics = {
        "run_name": run_name,
        "task": args.task,
        "model_name": args.model_name,
        "use_class_weights": args.use_class_weights,
        "use_fgm": args.use_fgm,
        "use_lora": args.use_lora,
        "trainable_params": trainable,
        "total_params": total,
        "seed": args.seed,
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "test_micro_f1": micro_f1,
        "per_class": {
            label_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1_per_class[i]),
                "support": int(support[i]),
            }
            for i in range(num_labels)
        },
        "confusion_matrix": cm.tolist(),
        "label_names": label_names,
    }

    metrics_path = os.path.join(run_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save raw predictions too -- needed later for McNemar's test between runs.
    np.save(os.path.join(run_dir, "test_preds.npy"), test_preds)
    np.save(os.path.join(run_dir, "test_labels.npy"), test_labels)

    print(f"[result] test_accuracy={acc:.4f} test_macro_f1={macro_f1:.4f}")
    print(f"[saved] {metrics_path}")


if __name__ == "__main__":
    main()
