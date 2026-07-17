"""
Post-hoc evaluation utilities:
  - confusion matrix plotting from a saved test_metrics.json
  - McNemar's test between two runs' saved predictions (test_preds.npy),
    for the "is the improvement statistically significant" defense question
  - soft-vote (probability-averaging, default) or hard-vote (majority) ensemble
    across N runs' saved predictions
  - label-shift correction (priorshift) for a single run, using the
    validation confusion matrix to correct for the train/test class-balance
    shift we diagnosed on this dataset -- see cmd_priorshift for details

Usage:
    python evaluate.py confusion --metrics_json ../runs/<run_name>/test_metrics.json
    python evaluate.py mcnemar --run_a ../runs/<run_a> --run_b ../runs/<run_b>
    python evaluate.py ensemble --runs ../runs/<run1> ../runs/<run2> ../runs/<run3> [--mode soft|hard]
    python evaluate.py priorshift --run ../runs/<run_name>
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


def cmd_priorshift(args):
    """Label-shift correction (Black-Box Shift Estimation, Lipton et al. 2018).

    We already established that the class balance differs between the
    train/validation pool and the real test set (e.g. Negative is ~47.6% of
    train/val but only ~42.7% of test for the 3-class task). BBSE corrects
    for exactly this without ever looking at test labels:

      1. Estimate the confusion matrix C[i,j] = P(predicted=i | true=j) from
         the validation set (uses val labels -- legitimate, val is not test).
      2. Compute the empirical distribution of the model's predictions on the
         *test inputs* (argmax of test_probs -- uses no test labels).
      3. Solve C @ p_test = q_hat for p_test, the estimated true test-label
         prior.
      4. Re-weight each test example's predicted-class probabilities by
         w[c] = p_test[c] / p_val[c], renormalize, and take the new argmax.

    test_labels.npy is only used afterwards, to report before/after metrics
    -- never as part of the correction itself.
    """
    val_probs = np.load(os.path.join(args.run, "val_probs.npy"))
    val_labels = np.load(os.path.join(args.run, "val_labels.npy"))
    test_probs = np.load(os.path.join(args.run, "test_probs.npy"))
    test_labels = np.load(os.path.join(args.run, "test_labels.npy"))

    num_labels = val_probs.shape[1]
    val_preds = np.argmax(val_probs, axis=-1)

    # Step 1: confusion matrix on validation, columns = true label.
    C = np.zeros((num_labels, num_labels))
    for true_c in range(num_labels):
        mask = val_labels == true_c
        if mask.sum() == 0:
            C[:, true_c] = 1.0 / num_labels
            continue
        for pred_c in range(num_labels):
            C[pred_c, true_c] = np.mean(val_preds[mask] == pred_c)

    p_val = np.array([np.mean(val_labels == c) for c in range(num_labels)])

    # Step 2: empirical predicted distribution on test inputs (no test labels used).
    test_preds_raw = np.argmax(test_probs, axis=-1)
    q_hat = np.array([np.mean(test_preds_raw == c) for c in range(num_labels)])

    # Step 3: solve C @ p = q_hat (least squares, then clip + renormalize to
    # keep it a valid probability distribution).
    p_test_est, *_ = np.linalg.lstsq(C, q_hat, rcond=None)
    p_test_est = np.clip(p_test_est, 1e-6, None)
    p_test_est = p_test_est / p_test_est.sum()

    print(f"[priorshift] validation class prior: {p_val.round(4).tolist()}")
    print(f"[priorshift] estimated test class prior: {p_test_est.round(4).tolist()}")

    # Step 4: re-weight and re-predict.
    w = p_test_est / p_val
    adjusted_probs = test_probs * w[None, :]
    adjusted_probs = adjusted_probs / adjusted_probs.sum(axis=-1, keepdims=True)
    adjusted_preds = np.argmax(adjusted_probs, axis=-1)

    before_acc = accuracy_score(test_labels, test_preds_raw)
    before_f1 = f1_score(test_labels, test_preds_raw, average="macro")
    after_acc = accuracy_score(test_labels, adjusted_preds)
    after_f1 = f1_score(test_labels, adjusted_preds, average="macro")

    print(f"[priorshift] before: accuracy={before_acc:.4f} macro_f1={before_f1:.4f}")
    print(f"[priorshift] after:  accuracy={after_acc:.4f} macro_f1={after_f1:.4f}")
    if after_f1 > before_f1:
        print(f"[priorshift] improved macro_f1 by {after_f1 - before_f1:+.4f}")
    else:
        print(f"[priorshift] did not improve macro_f1 ({after_f1 - before_f1:+.4f}) "
              "-- the estimated shift may be too noisy on this validation size; "
              "report the uncorrected number.")


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

    p_ps = sub.add_parser("priorshift")
    p_ps.add_argument("--run", required=True)
    p_ps.set_defaults(func=cmd_priorshift)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
