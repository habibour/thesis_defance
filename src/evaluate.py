"""
Post-hoc evaluation utilities:
  - confusion matrix plotting from a saved test_metrics.json
  - McNemar's test between two runs' saved predictions (test_preds.npy),
    for the "is the improvement statistically significant" defense question
  - soft-vote (probability-averaging, default) or hard-vote (majority) ensemble
    across N runs' saved predictions

Usage:
    python evaluate.py confusion --metrics_json ../runs/<run_name>/test_metrics.json
    python evaluate.py mcnemar --run_a ../runs/<run_a> --run_b ../runs/<run_b>
    python evaluate.py ensemble --runs ../runs/<run1> ../runs/<run2> ../runs/<run3> [--mode soft|hard]
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar


def cmd_confusion(args):
    import matplotlib.pyplot as plt
    import seaborn as sns

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["label_names"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Reds", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(metrics["run_name"])
    out_path = os.path.splitext(args.metrics_json)[0] + "_confusion.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}")
    print(f"accuracy={metrics['test_accuracy']:.4f} macro_f1={metrics['test_macro_f1']:.4f}")


def cmd_mcnemar(args):
    """Paired significance test between two models' predictions on the same
    test set: is run_b's improvement over run_a real, or noise?
    """
    preds_a = np.load(os.path.join(args.run_a, "test_preds.npy"))
    preds_b = np.load(os.path.join(args.run_b, "test_preds.npy"))
    labels_a = np.load(os.path.join(args.run_a, "test_labels.npy"))
    labels_b = np.load(os.path.join(args.run_b, "test_labels.npy"))

    assert np.array_equal(labels_a, labels_b), (
        "The two runs were evaluated on different label orderings/sets -- "
        "make sure both used the same test split before comparing."
    )
    labels = labels_a

    a_correct = preds_a == labels
    b_correct = preds_b == labels

    both_correct = int(np.sum(a_correct & b_correct))
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))
    table = [[both_correct, a_only], [b_only, both_wrong]]

    result = mcnemar(table, exact=(a_only + b_only) < 25, correction=True)

    print(f"run_a={args.run_a} acc={accuracy_score(labels, preds_a):.4f}")
    print(f"run_b={args.run_b} acc={accuracy_score(labels, preds_b):.4f}")
    print(f"contingency table: {table}")
    print(f"McNemar statistic={result.statistic:.4f}  p-value={result.pvalue:.6f}")
    if result.pvalue < 0.05:
        print("-> difference is statistically significant at alpha=0.05")
    else:
        print("-> difference is NOT statistically significant at alpha=0.05 "
              "(consider more seeds/epochs, or report honestly as inconclusive)")


def cmd_ensemble(args):
    """Ensemble across N runs' saved predictions.

    --mode soft (default): average each model's predicted-class probabilities
    (test_probs.npy) and argmax the result -- this is the confidence-summing
    approach BLP-2023's Knowdee system used, and is generally stronger than
    hard voting. Requires each run to have been produced by a train.py that
    saves test_probs.npy (current version does; older saved runs may not).

    --mode hard: majority vote over already-argmaxed predictions
    (test_preds.npy) -- the original, simpler fallback, usable even for
    older runs that predate probability saving.
    """
    labels = np.load(os.path.join(args.runs[0], "test_labels.npy"))

    if args.mode == "soft":
        missing = [r for r in args.runs if not os.path.exists(os.path.join(r, "test_probs.npy"))]
        if missing:
            raise FileNotFoundError(
                f"test_probs.npy missing for: {missing}. These runs predate probability "
                "saving -- rerun them, or use --mode hard instead."
            )
        all_probs = [np.load(os.path.join(r, "test_probs.npy")) for r in args.runs]
        for r, p in zip(args.runs, all_probs):
            assert len(p) == len(labels), f"{r} has a different test set size"
        avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)  # [n_examples, n_labels]
        votes = np.argmax(avg_probs, axis=-1)
    else:
        all_preds = [np.load(os.path.join(r, "test_preds.npy")) for r in args.runs]
        for r, p in zip(args.runs, all_preds):
            assert len(p) == len(labels), f"{r} has a different test set size"
        stacked = np.stack(all_preds, axis=0)  # [n_runs, n_examples]
        num_labels = int(stacked.max()) + 1
        votes = np.apply_along_axis(
            lambda col: np.bincount(col, minlength=num_labels).argmax(), axis=0, arr=stacked
        )

    acc = accuracy_score(labels, votes)
    macro_f1 = f1_score(labels, votes, average="macro")
    print(f"[ensemble of {len(args.runs)} runs, mode={args.mode}] accuracy={acc:.4f} macro_f1={macro_f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_conf = sub.add_parser("confusion")
    p_conf.add_argument("--metrics_json", required=True)
    p_conf.set_defaults(func=cmd_confusion)

    p_mcn = sub.add_parser("mcnemar")
    p_mcn.add_argument("--run_a", required=True)
    p_mcn.add_argument("--run_b", required=True)
    p_mcn.set_defaults(func=cmd_mcnemar)

    p_ens = sub.add_parser("ensemble")
    p_ens.add_argument("--runs", nargs="+", required=True)
    p_ens.add_argument("--mode", choices=["soft", "hard"], default="soft")
    p_ens.set_defaults(func=cmd_ensemble)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
