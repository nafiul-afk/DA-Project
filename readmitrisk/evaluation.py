"""Metrics and operating points, kept separate from model training."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve)


def compute_metrics(y, probabilities, threshold=.5):
    y, p = np.asarray(y).astype(int), np.asarray(probabilities).astype(float)
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(accuracy=float(accuracy_score(y, prediction)),
        balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
        precision=float(precision_score(y, prediction, zero_division=0)),
        recall=float(recall_score(y, prediction, zero_division=0)),
        f1=float(f1_score(y, prediction, zero_division=0)),
        roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        pr_auc=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)),
        threshold=float(threshold), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        n=len(y), prevalence=float(y.mean()), selection_rate=float(prediction.mean()))


def threshold_curve(y, probabilities):
    """Exact thresholds at every distinct probability, including no selection."""
    y, p = np.asarray(y).astype(int), np.asarray(probabilities)
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(ps) != 0), len(p)-1]
    tp = np.r_[0, np.cumsum(ys)[ends]]
    selected = np.r_[0, ends+1]
    fp = selected-tp
    positive, negative = y.sum(), len(y)-y.sum()
    return pd.DataFrame(dict(threshold=np.r_[np.nextafter(1., 2.), ps[ends]],
        accuracy=(tp+negative-fp)/len(y), recall=tp/max(positive, 1),
        precision=np.divide(tp, selected, out=np.zeros(len(tp), dtype=float), where=selected > 0),
        selection_rate=selected/len(y)))


def choose_thresholds(y, probabilities, accuracy_floor=.82, recall_goal=.70):
    """Lock thresholds on validation patients only; test labels never enter.

    82% validation accuracy leaves a predeclared two-point margin for the 80%
    requested test target. It is not a guarantee about unseen patients.
    """
    curve = threshold_curve(y, probabilities)
    feasible = curve.loc[curve.accuracy >= accuracy_floor]
    if len(feasible):
        balanced = feasible.sort_values(["recall", "accuracy"], ascending=False).iloc[0]
    else:
        balanced = curve.sort_values(["accuracy", "recall"], ascending=False).iloc[0]
    high = curve.loc[curve.recall >= recall_goal].sort_values(["precision", "accuracy"], ascending=False).iloc[0]
    return {"balanced": float(balanced.threshold), "high_recall": float(high.threshold)}


def decile_table(y, probabilities):
    frame = pd.DataFrame(dict(y=np.asarray(y), probability=np.asarray(probabilities)))
    frame = frame.sort_values("probability", ascending=False, kind="stable").reset_index(drop=True)
    frame["decile"] = np.minimum(np.arange(len(frame))*10 // len(frame)+1, 10)
    table = frame.groupby("decile").agg(encounters=("y", "size"), readmissions=("y", "sum"),
        observed_rate=("y", "mean"), mean_probability=("probability", "mean"))
    table["lift"] = table.observed_rate / frame.y.mean()
    table["cumulative_fraction"] = table.encounters.cumsum() / len(frame)
    table["cumulative_capture"] = table.readmissions.cumsum() / frame.y.sum()
    return table.reset_index()


def save_diagnostic_plots(y, probabilities, thresholds, output_dir=Path("."), prefix="classifier"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.calibration import calibration_curve
    root = Path(output_dir)
    (root / "figures").mkdir(exist_ok=True, parents=True)
    (root / "tables").mkdir(exist_ok=True, parents=True)
    y, p = np.asarray(y), np.asarray(probabilities)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, color="#138f87", label=f"ROC-AUC {roc_auc_score(y,p):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#9ca3af")
    axes[0].set(xlabel="False positive rate", ylabel="Recall", title="ROC curve")
    precision, recall, _ = precision_recall_curve(y, p)
    axes[1].plot(recall, precision, color="#138f87", label=f"AP / PR-AUC {average_precision_score(y,p):.3f}")
    axes[1].axhline(y.mean(), ls="--", color="#9ca3af", label=f"Prevalence {y.mean():.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve")
    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    axes[2].plot(predicted, observed, "o-", color="#138f87", label=f"Brier {brier_score_loss(y,p):.3f}")
    axes[2].plot([0, max(.4, observed.max())], [0, max(.4, observed.max())], "--", color="#9ca3af")
    axes[2].set(xlabel="Mean predicted probability", ylabel="Observed rate", title="Calibration (equal-count bins)")
    for ax in axes:
        ax.legend(fontsize=9)
        sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(root / f"figures/{prefix}_roc_pr_calibration.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(dict(mean_probability=predicted, observed_rate=observed)).to_csv(root / f"tables/{prefix}_calibration_bins.csv", index=False)
    fig, axes = plt.subplots(1, len(thresholds), figsize=(6*len(thresholds), 4), squeeze=False)
    for ax, (name, threshold) in zip(axes.ravel(), thresholds.items()):
        sns.heatmap(confusion_matrix(y, p >= threshold), annot=True, fmt="d", cmap="BuGn", cbar=False, ax=ax)
        ax.set(title=f"{name.replace('_',' ').title()} | threshold={threshold:.3f}", xlabel="Predicted", ylabel="Observed")
    fig.tight_layout()
    fig.savefig(root / f"figures/{prefix}_confusion.png", dpi=170)
    plt.close(fig)
    deciles = decile_table(y, p)
    deciles.to_csv(root / f"tables/{prefix}_deciles.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(deciles.decile, deciles.lift, color="#138f87")
    axes[0].axhline(1, ls="--", color="gray")
    axes[0].set(title="Lift by risk decile", xlabel="Decile (1 = highest risk)", ylabel="Readmission rate / prevalence")
    axes[1].plot(np.r_[0, deciles.cumulative_fraction], np.r_[0, deciles.cumulative_capture], "o-", color="#138f87")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(title="Cumulative capture", xlabel="Fraction of encounters selected", ylabel="Fraction of readmissions captured")
    fig.tight_layout()
    fig.savefig(root / f"figures/{prefix}_lift_gain.png", dpi=170)
    plt.close(fig)
    return deciles


def fairness_table(frame, probabilities, thresholds):
    """Descriptive audit, not a causal fairness guarantee; show denominators."""
    rows = []
    for operating_point, threshold in thresholds.items():
        for attribute in ["race", "gender", "age"]:
            for level, positions in frame.reset_index(drop=True).groupby(attribute).groups.items():
                positions = np.asarray(list(positions))
                target = frame.iloc[positions].readmit_30.to_numpy()
                pred = np.asarray(probabilities)[positions] >= threshold
                tp = int(((target == 1) & pred).sum())
                positive = int(target.sum())
                # Wilson interval for recall: unstable small groups remain visible.
                recall = tp/positive if positive else np.nan
                z = 1.96
                if positive:
                    center = (recall + z*z/(2*positive))/(1+z*z/positive)
                    half = z*np.sqrt(recall*(1-recall)/positive+z*z/(4*positive**2))/(1+z*z/positive)
                else:
                    center = half = np.nan
                rows.append(dict(operating_point=operating_point, attribute=attribute, group=str(level),
                    n=len(positions), positives=positive, recall=recall,
                    recall_low=center-half, recall_high=center+half,
                    selection_rate=float(pred.mean()), prevalence=float(target.mean()),
                    small_group=len(positions)<100 or positive<20))
    return pd.DataFrame(rows)
