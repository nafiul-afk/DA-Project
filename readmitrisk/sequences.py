"""Patient history forecasting with train-only preparation and grouped evaluation.

The target is the *next encounter's* 30-day readmission outcome.  Only encounters
strictly preceding that target encounter enter a history.  ``encounter_id`` is a
time-order proxy, not an input feature.  The history baseline observes exactly
the same records as the GRU and LSTM.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
HISTORY_LENGTH = 3
EXCLUDED_COLUMNS = {
    "encounter_id", "patient_nbr", "readmitted", "readmit_30", "target",
    "row_id", "row_index", "index", "split", "weight",
    "age", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "diag_1", "diag_2", "diag_3",
}


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _metric_suite(y, probabilities, threshold=0.5, predictions=None):
    """Natural-distribution metrics; optional per-fold operating decisions."""
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int) if predictions is None else predictions
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if both_classes else None,
        "pr_auc": float(average_precision_score(y, probabilities)) if both_classes else None,
        "brier": float(brier_score_loss(y, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _thresholds(y, probabilities, minimum_accuracy=0.82, target_recall=0.70):
    """Select both operating points on validation labels only.

    A modest accuracy margin is requested on validation, but never guaranteed
    on test. If the requested validation accuracy is infeasible, use the best
    validation accuracy and report the resulting shortfall.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    order = np.argsort(p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    candidates = np.r_[0.0, np.unique(sorted_p), 1.000001]
    below = np.searchsorted(sorted_p, candidates, side="left")
    cumulative = np.r_[0, np.cumsum(sorted_y)]
    fn = cumulative[below]
    tp = y.sum() - fn
    tn = below - fn
    accuracy = (tn + tp) / len(y)
    recall = tp / max(int(y.sum()), 1)
    feasible = np.flatnonzero(accuracy >= minimum_accuracy)
    if len(feasible):
        # Ascending threshold => maximum recall among feasible thresholds.
        chosen = feasible[0]
    else:
        chosen = int(np.argmax(accuracy))
    recall_feasible = np.flatnonzero(recall >= target_recall)
    high = recall_feasible[-1] if len(recall_feasible) else 0
    return {"balanced": float(candidates[chosen]), "high_recall": float(candidates[high])}


class SequenceSigmoidCalibrator:
    """Platt scaling on patient-disjoint calibration data."""

    def fit(self, probabilities, y):
        self.model_ = LogisticRegression(C=1e3, solver="lbfgs", random_state=SEED)
        self.model_.fit(self._logit(probabilities), np.asarray(y, dtype=int))
        return self

    @staticmethod
    def _logit(probabilities):
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict(self, probabilities):
        return self.model_.predict_proba(self._logit(probabilities))[:, 1]


def make_history_index(df, split_indices):
    """Return fixed-length prior-row indices, with -1 denoting left padding.

    ``split_indices`` refer to integer row positions in ``df``, as produced by
    sklearn splitters. Group membership is checked before histories are built.
    """
    required = {"encounter_id", "patient_nbr", "readmit_30"}
    if not required.issubset(df.columns):
        raise ValueError(f"Sequence data require metadata columns: {sorted(required)}")
    frame = df.reset_index(drop=True)
    split_of_row = np.full(len(frame), "", dtype=object)
    seen_patients = set()
    for split, indices in split_indices.items():
        indices = np.asarray(indices, dtype=int)
        patients = set(frame.iloc[indices]["patient_nbr"].tolist())
        if seen_patients & patients:
            raise ValueError("A patient occurs in more than one sequence split.")
        if np.any(split_of_row[indices] != ""):
            raise ValueError("Sequence split indices overlap.")
        seen_patients.update(patients)
        split_of_row[indices] = split
    if np.any(split_of_row == ""):
        raise ValueError("Every cleaned encounter must belong to a split.")
    ordered = pd.DataFrame({
        "row": np.arange(len(frame)),
        "patient": frame["patient_nbr"].to_numpy(),
        "encounter": pd.to_numeric(frame["encounter_id"], errors="raise").to_numpy(),
    }).sort_values(["patient", "encounter", "row"], kind="stable")
    history_rows, target_rows, lengths = [], [], []
    for _, records in ordered.groupby("patient", sort=False):
        rows = records["row"].to_numpy(dtype=int)
        encounters = records["encounter"].to_numpy()
        for position in range(1, len(rows)):
            previous = rows[max(0, position - HISTORY_LENGTH):position]
            # Duplicate encounter identifiers cannot establish chronological order.
            if encounters[position - 1] >= encounters[position]:
                raise ValueError("Encounter identifiers must increase within a patient.")
            padded = np.full(HISTORY_LENGTH, -1, dtype=int)
            padded[-len(previous):] = previous
            history_rows.append(padded)
            target_rows.append(rows[position])
            lengths.append(len(previous))
    histories = np.asarray(history_rows, dtype=int)
    targets = np.asarray(target_rows, dtype=int)
    if not len(targets):
        raise ValueError("No patient has at least two eligible encounters.")
    sequence_splits = {
        split: np.flatnonzero(split_of_row[targets] == split) for split in split_indices
    }
    for split, indices in sequence_splits.items():
        if not len(indices):
            raise ValueError(f"No sequence-eligible targets in split {split}.")
        used_rows = histories[indices].ravel()
        used_rows = used_rows[used_rows >= 0]
        if not np.all(split_of_row[used_rows] == split):
            raise AssertionError("A history crosses patient split boundaries.")
    return histories, targets, np.asarray(lengths), sequence_splits


def _feature_frame(df):
    features = [
        col for col in df.columns if col not in EXCLUDED_COLUMNS
        and "readmit" not in col.lower() and not col.lower().startswith("target")
    ]
    if not features:
        raise ValueError("No permitted encounter features remain.")
    frame = df[features].copy()
    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [col for col in features if col not in numeric]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    for col in categorical:
        frame[col] = frame[col].astype("object").where(frame[col].notna(), "Unknown").astype(str)
    return frame, numeric, categorical


def _preprocessor(numeric, categorical):
    # Infrequent category grouping is learned only from training histories.
    from .data import NearZeroDropper, RareCategoryGrouper
    encoding = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), make_column_selector(dtype_include=np.number)),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=0.01, max_categories=30,
                                      sparse_output=False, dtype=np.float32)),
        ]), make_column_selector(dtype_exclude=np.number)),
    ], remainder="drop", sparse_threshold=0)
    return Pipeline([("near_zero", NearZeroDropper()),
                     ("rare", RareCategoryGrouper()), ("encode", encoding)])


def _fit_prepare(frame, numeric, categorical, histories, training_sequences, sequences):
    used_rows = np.unique(histories[training_sequences])
    used_rows = used_rows[used_rows >= 0]
    preprocessing = _preprocessor(numeric, categorical)
    preprocessing.fit(frame.iloc[used_rows])
    relevant_rows = np.unique(histories[sequences])
    relevant_rows = relevant_rows[relevant_rows >= 0]
    # Transform each encounter once rather than expanding a large repeated frame.
    encoded = np.zeros((len(frame) + 1, len(preprocessing.get_feature_names_out())), dtype=np.float32)
    encoded[relevant_rows + 1] = preprocessing.transform(frame.iloc[relevant_rows]).astype(np.float32)
    tensor = encoded[histories[sequences] + 1]
    return preprocessing, tensor


def _baseline_features(tensor, lengths):
    """Last encounter + mean of observed history + number of observed encounters."""
    means = tensor.sum(axis=1) / np.asarray(lengths).reshape(-1, 1)
    return np.column_stack([tensor[:, -1, :], means, lengths]).astype(np.float32)


def _tensorflow():
    # Limit parallelism before importing TensorFlow; CPU-only makes runs portable.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/readmitrisk-matplotlib")
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass  # The notebook may already have initialized TensorFlow.
    return tf


def _dataset(tf, x, y=None, batch_size=256, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(x if y is None else (x, y))
    if shuffle:
        dataset = dataset.shuffle(min(len(x), 20000), seed=SEED, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.batch(batch_size).with_options(options).prefetch(1)


def _train_recurrent(model_name, x_train, y_train, x_validation, y_validation, *, epochs, seed):
    tf = _tensorflow()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    recurrent = tf.keras.layers.GRU if model_name == "GRU" else tf.keras.layers.LSTM
    inputs = tf.keras.Input(shape=x_train.shape[1:], name="prior_encounters")
    x = tf.keras.layers.Masking(mask_value=0.0, name="padding_mask")(inputs)
    x = recurrent(32, dropout=0.20, name=model_name.lower())(x)
    x = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="next_encounter_risk")(x)
    model = tf.keras.Model(inputs, outputs, name=f"readmitrisk_{model_name.lower()}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    classes = np.bincount(y_train.astype(int), minlength=2)
    weights = {label: len(y_train) / (2 * max(int(count), 1)) for label, count in enumerate(classes)}
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=3,
                                         restore_best_weights=True, min_delta=0.001),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", patience=2,
                                             factor=0.5, min_lr=1e-5),
    ]
    history = model.fit(_dataset(tf, x_train, y_train, shuffle=True),
                        validation_data=_dataset(tf, x_validation, y_validation),
                        epochs=epochs, callbacks=callbacks, class_weight=weights, shuffle=False, verbose=0)
    return model, history.history


def _predict_recurrent(model, tensor):
    return model.predict(_dataset(_tensorflow(), tensor), verbose=0).reshape(-1)


def _train_baseline(x, y, seed=SEED):
    from xgboost import XGBClassifier
    count = np.bincount(np.asarray(y, dtype=int), minlength=2)
    model = XGBClassifier(n_estimators=220, max_depth=3, learning_rate=0.045,
                          min_child_weight=8, subsample=0.85, colsample_bytree=0.8,
                          reg_lambda=8.0, reg_alpha=0.2,
                          scale_pos_weight=float(count[0] / max(count[1], 1)),
                          objective="binary:logistic", eval_metric="aucpr",
                          tree_method="hist", n_jobs=2, random_state=seed)
    model.fit(x, y)
    return model


def _inner_group_split(indices, labels, groups, seed):
    # All three internal splits are patient-disjoint; outer validation is untouched.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    fit_rel, hold_rel = next(outer.split(indices, labels[indices], groups[indices]))
    fit, hold = indices[fit_rel], indices[hold_rel]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    cal_rel, val_rel = next(inner.split(hold, labels[hold], groups[hold]))
    return fit, hold[cal_rel], hold[val_rel]


def _run_grouped_cv(frame, numeric, categorical, histories, lengths, y, groups,
                    training_sequences, output_dir, cv_epochs):
    """Actual five-fold grouped CV with fold-local preparation/calibration/tuning."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, prediction_rows, history_rows = [], [], []
    model_names = ["GRU", "LSTM", "XGBoost history"]
    for fold, (fit_rel, score_rel) in enumerate(
        splitter.split(training_sequences, y[training_sequences], groups[training_sequences]), start=1
    ):
        fold_pool, score = training_sequences[fit_rel], training_sequences[score_rel]
        fit, calibrate, validate = _inner_group_split(fold_pool, y, groups, SEED + fold)
        together = np.concatenate([fit, calibrate, validate, score])
        _, tensor = _fit_prepare(frame, numeric, categorical, histories, fit, together)
        cut1, cut2, cut3 = len(fit), len(fit) + len(calibrate), len(fit) + len(calibrate) + len(validate)
        partitions = {
            "train": tensor[:cut1], "calibration": tensor[cut1:cut2],
            "validation": tensor[cut2:cut3], "score": tensor[cut3:],
        }
        index_parts = {"train": fit, "calibration": calibrate, "validation": validate, "score": score}
        for model_name in model_names:
            start = time.perf_counter()
            if model_name == "XGBoost history":
                matrices = {key: _baseline_features(value, lengths[index_parts[key]])
                            for key, value in partitions.items()}
                model = _train_baseline(matrices["train"], y[fit], SEED + fold)
                raw = {key: model.predict_proba(value)[:, 1] for key, value in matrices.items() if key != "train"}
            else:
                model, history = _train_recurrent(
                    model_name, partitions["train"], y[fit], partitions["validation"], y[validate],
                    epochs=cv_epochs, seed=SEED + fold,
                )
                for epoch in range(len(history["loss"])):
                    history_rows.append({"model": model_name, "run": f"cv_{fold}", "epoch": epoch + 1,
                                         **{key: value[epoch] for key, value in history.items()}})
                raw = {key: _predict_recurrent(model, value) for key, value in partitions.items() if key != "train"}
            calibrator = SequenceSigmoidCalibrator().fit(raw["calibration"], y[calibrate])
            val_p, score_p = calibrator.predict(raw["validation"]), calibrator.predict(raw["score"])
            operating = _thresholds(y[validate], val_p)
            for point, threshold in operating.items():
                rows.append({"model": model_name, "fold": fold, "operating_point": point,
                             "threshold": threshold, "n": len(score),
                             "training_seconds": time.perf_counter() - start,
                             **_metric_suite(y[score], score_p, threshold)})
            prediction_rows.extend({
                "model": model_name, "fold": fold, "sequence_row": int(index), "y_true": int(label),
                "probability": float(probability),
                "balanced_prediction": int(probability >= operating["balanced"]),
                "high_recall_prediction": int(probability >= operating["high_recall"]),
            } for index, label, probability in zip(score, y[score], score_p))
            del model, raw
            gc.collect()
            print(f"  Sequence CV fold {fold}/5: {model_name} finished", flush=True)
        del tensor, partitions
        gc.collect()
    scores = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    scores.to_csv(output_dir / "tables/sequence_cv_scores.csv", index=False)
    predictions.to_csv(output_dir / "tables/sequence_cv_predictions.csv", index=False)
    columns = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
    summary = scores.groupby(["model", "operating_point"])[columns].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output_dir / "tables/sequence_cv_summary.csv", index=False)
    pooled = []
    for model_name, records in predictions.groupby("model", sort=False):
        for point in ["balanced", "high_recall"]:
            pooled.append({"model": model_name, "split": "cv", "operating_point": point,
                           "threshold": None, "n": len(records),
                           **_metric_suite(records.y_true, records.probability,
                                           predictions=records[f"{point}_prediction"].to_numpy())})
    return rows, pooled, history_rows


def _save_figures(predictions, comparison, histories, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 160, "figure.facecolor": "white"})
    colors = {"GRU": "#008b8b", "LSTM": "#205c70", "XGBoost history": "#e8a23a"}
    test = predictions.loc[predictions["split"] == "test"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    gains = []
    for name, records in test.groupby("model", sort=False):
        color = colors[name]
        RocCurveDisplay.from_predictions(records.y_true, records.probability, name=name, ax=axes[0, 0], color=color)
        PrecisionRecallDisplay.from_predictions(records.y_true, records.probability, name=name, ax=axes[0, 1], color=color)
        observed, predicted = calibration_curve(records.y_true, records.probability, n_bins=8, strategy="quantile")
        axes[1, 0].plot(predicted, observed, "o-", label=name, color=color)
        ranked = records.sort_values("probability", ascending=False).reset_index(drop=True)
        cumulative = ranked.y_true.cumsum().to_numpy() / max(int(ranked.y_true.sum()), 1)
        fraction = np.arange(1, len(ranked) + 1) / len(ranked)
        axes[1, 1].plot(fraction, cumulative, label=name, color=color)
        for decile in range(1, 11):
            n = int(np.ceil(len(ranked) * decile / 10))
            captured = float(ranked.y_true.iloc[:n].sum() / max(int(ranked.y_true.sum()), 1))
            gains.append({"model": name, "decile": decile, "fraction_scored": n / len(ranked),
                          "captured_readmissions": captured, "cumulative_lift": captured / (n / len(ranked)),
                          "n": n})
    axes[0, 0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0, 0].set_title("Held-out ROC: same eligible patients")
    axes[0, 1].axhline(test.y_true.mean(), ls="--", color="gray", linewidth=1)
    axes[0, 1].set_title("Held-out precision–recall")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[1, 0].set(xlabel="Calibrated forecast probability", ylabel="Observed readmission rate", title="Calibration")
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[1, 1].set(xlabel="Fraction of encounters prioritized", ylabel="Fraction of readmissions captured", title="Cumulative gain")
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.suptitle("Sequential forecasting: only earlier encounter features", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "figures/sequence_diagnostics.png", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(gains).to_csv(output_dir / "tables/sequence_lift_gain.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, name in enumerate(colors):
        records = test.loc[test.model == name]
        for row, point in enumerate(["balanced", "high_recall"]):
            matrix = confusion_matrix(records.y_true, records[f"{point}_prediction"], labels=[0, 1])
            ax = axes[row, col]
            ax.imshow(matrix, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                            color="white" if matrix[i, j] > matrix.max() / 2 else "#123")
            ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual",
                   title=f"{name} · {point.replace('_', ' ')}")
    fig.suptitle("Sequence held-out confusion matrices", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "figures/sequence_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)

    history_df = pd.DataFrame(histories)
    history_df.to_csv(output_dir / "tables/sequence_training_history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ["GRU", "LSTM"]:
        records = history_df.loc[(history_df.model == name) & (history_df.run == "final")]
        axes[0].plot(records.epoch, records.loss, label=f"{name} train", color=colors[name])
        axes[0].plot(records.epoch, records.val_loss, "--", label=f"{name} validation", color=colors[name])
        axes[1].plot(records.epoch, records.pr_auc, label=f"{name} train", color=colors[name])
        axes[1].plot(records.epoch, records.val_pr_auc, "--", label=f"{name} validation", color=colors[name])
    axes[0].set(title="Training curves (class-weighted training loss)", xlabel="Epoch", ylabel="Binary cross-entropy")
    axes[1].set(title="PR-AUC training curves", xlabel="Epoch", ylabel="Keras approximate PR-AUC")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "figures/sequence_training_curves.png", bbox_inches="tight")
    plt.close(fig)


def run_sequences(df, split_indices, output_dir=Path("."), *, epochs=15, cv_epochs=8):
    """Train, save, and evaluate GRU/LSTM/history-XGBoost forecasting agents.

    Five-fold CV runs on the training-patient partition. Each fold owns an
    internal training, calibration, and validation split. Reported held-out test
    metrics are computed once after model fitting and threshold selection.
    """
    output_dir = Path(output_dir)
    for directory in ["artifacts", "tables", "figures"]:
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    frame = df.reset_index(drop=True)
    histories, targets, lengths, sequence_splits = make_history_index(frame, split_indices)
    for required in ["train", "calibration", "validation", "test"]:
        if required not in sequence_splits:
            raise ValueError(f"Required patient split is missing: {required}")
    y = frame.iloc[targets]["readmit_30"].to_numpy(dtype=np.int32)
    groups = frame.iloc[targets]["patient_nbr"].to_numpy()
    features, numeric, categorical = _feature_frame(frame)
    count_rows = [{"split": split, "sequences": len(indices),
                   "eligible_patients": len(np.unique(groups[indices])),
                   "readmissions": int(y[indices].sum()), "readmission_rate": float(y[indices].mean()),
                   "mean_history_length": float(lengths[indices].mean())}
                  for split, indices in sequence_splits.items()]
    pd.DataFrame(count_rows).to_csv(output_dir / "tables/sequence_sample_counts.csv", index=False)
    audit = pd.DataFrame({
        "sequence_row": np.arange(len(targets)), "target_row": targets,
        "patient_nbr": groups, "target_encounter_id": frame.iloc[targets].encounter_id.to_numpy(),
        "last_history_encounter_id": frame.iloc[histories[:, -1]].encounter_id.to_numpy(),
        "history_length": lengths, "y_true": y,
    })
    audit["history_row_indices"] = ["|".join(str(i) for i in row if i >= 0) for row in histories]
    audit["split"] = ""
    for split, indices in sequence_splits.items():
        audit.loc[indices, "split"] = split
    audit.to_csv(output_dir / "tables/sequence_history_audit.csv", index=False)
    print(f"Sequence cohort: {len(y):,} next-encounter targets, {len(np.unique(groups)):,} patients; "
          f"natural readmission rate {y.mean():.2%}.", flush=True)

    cv_rows, comparison_rows, training_history = _run_grouped_cv(
        features, numeric, categorical, histories, lengths, y, groups,
        sequence_splits["train"], output_dir, cv_epochs,
    )
    preprocessing, tensor = _fit_prepare(features, numeric, categorical, histories,
                                         sequence_splits["train"], np.arange(len(y)))
    partitions = {split: tensor[indices] for split, indices in sequence_splits.items()}
    baseline = {split: _baseline_features(value, lengths[sequence_splits[split]])
                for split, value in partitions.items()}
    joblib.dump({"preprocessor": preprocessing, "feature_columns": features.columns.tolist(),
                 "numeric_features": numeric, "categorical_features": categorical,
                 "history_length": HISTORY_LENGTH, "padding_value": 0.0,
                 "forecast_definition": "Prior encounters predict next encounter's readmit_30; target encounter excluded."},
                output_dir / "artifacts/sequence_preprocessor.joblib")
    prediction_rows, model_results = [], {}
    for model_name in ["GRU", "LSTM", "XGBoost history"]:
        start = time.perf_counter()
        if model_name == "XGBoost history":
            model = _train_baseline(baseline["train"], y[sequence_splits["train"]])
            raw = {split: model.predict_proba(value)[:, 1] for split, value in baseline.items()}
            joblib.dump(model, output_dir / "artifacts/sequence_xgboost.joblib")
            trained_epochs = None
        else:
            model, history = _train_recurrent(
                model_name, partitions["train"], y[sequence_splits["train"]],
                partitions["validation"], y[sequence_splits["validation"]], epochs=epochs, seed=SEED,
            )
            trained_epochs = len(history["loss"])
            for epoch in range(trained_epochs):
                training_history.append({"model": model_name, "run": "final", "epoch": epoch + 1,
                                         **{key: values[epoch] for key, values in history.items()}})
            raw = {split: _predict_recurrent(model, value) for split, value in partitions.items()}
            model.save(output_dir / f"artifacts/{model_name.lower()}.keras")
        training_seconds = time.perf_counter() - start
        calibrator = SequenceSigmoidCalibrator().fit(raw["calibration"], y[sequence_splits["calibration"]])
        probabilities = {split: calibrator.predict(value) for split, value in raw.items()}
        operating = _thresholds(y[sequence_splits["validation"]], probabilities["validation"])
        file_stem = model_name.lower().replace(" ", "_")
        joblib.dump(calibrator, output_dir / f"artifacts/sequence_{file_stem}_calibrator.joblib")
        result = {"thresholds": operating, "training_seconds": training_seconds,
                  "epochs_trained": trained_epochs, "metrics": {}}
        for split, indices in sequence_splits.items():
            result["metrics"][split] = {}
            for point, threshold in operating.items():
                metrics = _metric_suite(y[indices], probabilities[split], threshold)
                result["metrics"][split][point] = metrics
                comparison_rows.append({"model": model_name, "split": split, "operating_point": point,
                                        "threshold": threshold, "n": len(indices),
                                        "training_seconds": training_seconds,
                                        "encoded_history_features": tensor.shape[-1], **metrics})
            prediction_rows.extend({
                "model": model_name, "split": split, "sequence_row": int(index),
                "target_row": int(targets[index]), "patient_nbr": int(groups[index]),
                "y_true": int(y[index]), "probability": float(probability),
                "balanced_prediction": int(probability >= operating["balanced"]),
                "high_recall_prediction": int(probability >= operating["high_recall"]),
            } for index, probability in zip(indices, probabilities[split]))
        result["test_accuracy_requirement_met"] = result["metrics"]["test"]["balanced"]["accuracy"] >= 0.80
        result["test_high_recall_requirement_met"] = result["metrics"]["test"]["high_recall"]["recall"] >= 0.60
        model_results[model_name] = result
        print(f"  {model_name}: test accuracy {result['metrics']['test']['balanced']['accuracy']:.3f}, "
              f"recall {result['metrics']['test']['balanced']['recall']:.3f}, "
              f"PR-AUC {result['metrics']['test']['balanced']['pr_auc']:.3f}", flush=True)
        del model, raw
        gc.collect()
    comparison = pd.DataFrame(comparison_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    comparison.to_csv(output_dir / "tables/sequence_model_comparison.csv", index=False)
    prediction_frame.to_csv(output_dir / "tables/sequence_predictions.csv", index=False)
    _save_figures(prediction_frame, comparison, training_history, output_dir)
    # Selection is made using validation probabilities, never held-out test scores.
    chosen_model = max(model_results, key=lambda name: model_results[name]["metrics"]["validation"]["balanced"]["pr_auc"])
    result = {
        "target": "readmit_30 of the next observed eligible encounter",
        "history_length": HISTORY_LENGTH, "ordering": "encounter_id (proxy, no timestamps)",
        "history_only": True, "patient_disjoint_splits": True,
        "input_features": features.columns.tolist(), "encoded_features": int(tensor.shape[-1]),
        "architecture": {"recurrent_units": 32, "dense_units": 16, "recurrent_dropout": 0.20,
                         "dense_dropout": 0.25, "mask_value": 0.0, "max_epochs": epochs,
                         "batch_size": 256, "early_stopping_patience": 3,
                         "reduce_lr_patience": 2, "class_weight": "balanced inverse frequency"},
        "calibration": "Sigmoid fitted on separate calibration patients",
        "threshold_selection": {"split": "validation", "requested_accuracy": 0.82,
                                "requested_high_recall": 0.70,
                                "heldout_accuracy_requirement": 0.80},
        "cross_validation": {"folds": 5, "population": "training patients only",
                             "recurrent_epoch_cap": cv_epochs,
                             "method": "StratifiedGroupKFold; fold-local preparation, internal calibration and threshold validation",
                             "scores": cv_rows},
        "sample_counts": count_rows, "models": model_results,
        "selected_by_validation_pr_auc": chosen_model,
        "limitations": [
            "Encounter identifiers approximate chronology; elapsed time between encounters is unavailable.",
            "Only people with a subsequent observed encounter are eligible, so this cohort is selected and higher-risk.",
            "The forecast predicts the next encounter's outcome, not whether or when a new encounter will occur.",
            "Inputs exclude the target encounter entirely; its contemporaneous clinical status is unknown at forecast time.",
            "Within-patient target histories may overlap, but complete patients remain within one split.",
            "Validation accuracy and recall targets do not guarantee the same held-out operating characteristics.",
            "Five-fold recurrent CV uses a shorter epoch cap than the final models to bound CPU runtime.",
        ],
    }
    (output_dir / "artifacts/sequence_metrics.json").write_text(
        json.dumps(result, indent=2, default=_json_value), encoding="utf-8"
    )
    return result
