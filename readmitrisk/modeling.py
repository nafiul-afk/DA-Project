"""Grouped classification, imbalance experiments, calibration and explanations.

Every statistical transform and sampler lives in an imblearn Pipeline. The test
cohort is scored only after estimator, feature set and thresholds are locked.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold, cross_validate
from xgboost import XGBClassifier

from .data import (MEDICATIONS, SEED, NearZeroDropper, RareCategoryGrouper,
                   feature_columns, make_preprocessor, save_json)
from .evaluation import (choose_thresholds, compute_metrics, fairness_table,
                         save_diagnostic_plots, threshold_curve)


def make_pipeline(estimator, sampler=None):
    return Pipeline([
        ("drop", NearZeroDropper()), ("rare", RareCategoryGrouper()),
        ("preprocess", make_preprocessor()),
        ("sampler", sampler if sampler is not None else "passthrough"),
        ("model", estimator),
    ])


def grouped_cv(X, y, groups):
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))
    for train, validation in splits:
        assert set(np.asarray(groups)[train]).isdisjoint(np.asarray(groups)[validation])
    return splits


def _cv_summary(pipeline, X, y, folds):
    result = cross_validate(pipeline, X, y, cv=folds, n_jobs=1, error_score="raise",
        scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc", "accuracy": "accuracy", "recall": "recall"})
    return dict(cv_pr_auc_mean=float(result["test_pr_auc"].mean()),
        cv_pr_auc_std=float(result["test_pr_auc"].std()),
        cv_roc_auc_mean=float(result["test_roc_auc"].mean()),
        cv_accuracy=float(result["test_accuracy"].mean()), cv_recall=float(result["test_recall"].mean()),
        cv_fit_seconds=float(result["fit_time"].sum()))


def transform_for_explanation(pipeline, X):
    """Inference skips resampling; never transform through a fit_resample call."""
    frame = pipeline.named_steps["drop"].transform(X)
    frame = pipeline.named_steps["rare"].transform(frame)
    encoded = pipeline.named_steps["preprocess"].transform(frame)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return np.asarray(encoded), np.asarray(names)


def _original_feature(encoded_name, columns):
    text = str(encoded_name).split("__", 1)[-1]
    for column in sorted(columns, key=len, reverse=True):
        if text == column or text.startswith(column + "_"):
            return column
    return text


def medication_redundancy(train, root):
    """Binary active flags; constants are reported separately rather than inverted."""
    flags = train[MEDICATIONS].ne("No").astype(float)
    active = flags.loc[:, flags.var() > 0]
    corr = active.corr()
    corr.to_csv(root / "tables/medication_correlation.csv", index_label="feature")
    z = (active-active.mean())/active.std(ddof=0)
    rows = []
    for column in MEDICATIONS:
        if column not in active:
            rows.append(dict(feature=column, vif=None, status="constant; removed"))
            continue
        y = z[column].to_numpy()
        other = z.drop(columns=column).to_numpy()
        coefficients, *_ = np.linalg.lstsq(other, y, rcond=None)
        residual_fraction = float(np.mean((y-other@coefficients)**2))
        rows.append(dict(feature=column, vif=float(1/max(residual_fraction, 1e-12)), status="estimated"))
    pd.DataFrame(rows).to_csv(root / "tables/medication_vif.csv", index=False)
    pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(corr) for b in corr.columns[i+1:] if abs(corr.loc[a, b]) > .95]
    pd.DataFrame(pairs, columns=["feature_a", "feature_b", "correlation"]).to_csv(root / "tables/redundant_medication_pairs.csv", index=False)
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="vlag", ax=ax, square=True)
    ax.set_title("Medication active flags: correlation on training patients")
    fig.tight_layout()
    fig.savefig(root / "figures/medication_correlation.png", dpi=150)
    plt.close(fig)
    return pairs


def explain_and_select(pipeline, xgb_pipeline, X_train, X_validation, y_validation, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    sample = X_train.sample(min(700, len(X_train)), random_state=SEED)
    encoded, encoded_names = transform_for_explanation(pipeline, sample)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    values = explainer(encoded, check_additivity=False)
    # RF returns one output per class; boosting returns positive-class log odds.
    if values.values.ndim == 3:
        values = values[:, :, 1]
    values.feature_names = encoded_names.tolist()
    mapping = [_original_feature(name, X_train.columns) for name in encoded_names]
    encoded_table = pd.DataFrame(dict(encoded_feature=encoded_names, feature=mapping,
        mean_abs_shap=np.abs(values.values).mean(axis=0)))
    encoded_table.to_csv(root / "tables/encoded_shap_importance.csv", index=False)
    importance = encoded_table.groupby("feature", as_index=False).mean_abs_shap.sum().sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(root / "tables/feature_importance.csv", index=False)
    joblib.dump(explainer, root / "artifacts/shap_explainer.joblib", compress=3)
    for kind in ["summary", "bar"]:
        plt.figure()
        shap.summary_plot(values.values, encoded, feature_names=encoded_names,
            plot_type="dot" if kind == "summary" else "bar", show=False, max_display=18)
        plt.tight_layout()
        plt.savefig(root / f"figures/shap_{kind}.png", dpi=170, bbox_inches="tight")
        plt.close("all")
    top = int(np.argmax(np.abs(values.values).mean(axis=0)))
    shap.dependence_plot(top, values.values, encoded, feature_names=encoded_names, show=False, interaction_index=None)
    plt.tight_layout()
    plt.savefig(root / "figures/shap_dependence.png", dpi=170, bbox_inches="tight")
    plt.close("all")
    for i in range(3):
        shap.plots.waterfall(values[i], show=False, max_display=10)
        plt.savefig(root / f"figures/shap_waterfall_patient_{i+1}.png", dpi=170, bbox_inches="tight")
        plt.close("all")
    # Held-out validation importance is diagnostic; top-18 selection uses only TRAIN SHAP.
    selection = np.random.default_rng(SEED).choice(len(X_validation), min(2500, len(X_validation)), replace=False)
    permutation = permutation_importance(pipeline, X_validation.iloc[selection], np.asarray(y_validation)[selection],
        scoring="average_precision", n_repeats=3, random_state=SEED, n_jobs=1)
    pd.DataFrame(dict(feature=X_validation.columns, importance=permutation.importances_mean,
        importance_std=permutation.importances_std)).sort_values("importance", ascending=False).to_csv(root / "tables/permutation_importance.csv", index=False)
    _, xgb_names = transform_for_explanation(xgb_pipeline, X_train.iloc[:2])
    gain = xgb_pipeline.named_steps["model"].get_booster().get_score(importance_type="gain")
    gain_rows = [dict(encoded_feature=name, feature=_original_feature(name, X_train.columns), gain=float(gain.get(f"f{i}", gain.get(name, 0)))) for i, name in enumerate(xgb_names)]
    pd.DataFrame(gain_rows).groupby("feature", as_index=False).gain.sum().sort_values("gain", ascending=False).to_csv(root / "tables/xgboost_gain.csv", index=False)
    pairs = medication_redundancy(X_train, root)
    removed = set()
    ranks = {c: i for i, c in enumerate(importance.feature)}
    for a, b, _ in pairs:
        removed.add(a if ranks.get(a, 999) > ranks.get(b, 999) else b)
    selected = [c for c in importance.feature if c not in removed][:18]
    save_json(root / "artifacts/selected_features.json", dict(features=selected, correlated_dropped=sorted(removed),
        selection_scope="Training patients only: sum of absolute one-hot SHAP importance per original feature.",
        explanation_units="Base estimator raw margin (log odds for boosting); not calibrated probability contributions."))
    return selected, importance


def _calibrate(pipeline, X, y):
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
    calibrated.fit(X, y)
    return calibrated


def save_deployed_explanations(pipeline, X_train, root):
    """Three named local explanations of the model actually used by the app."""
    import matplotlib.pyplot as plt
    import shap
    sample = X_train.sample(3, random_state=SEED)
    encoded, names = transform_for_explanation(pipeline, sample)
    explanations = shap.TreeExplainer(pipeline.named_steps["model"])(encoded, check_additivity=False)
    if explanations.values.ndim == 3:
        explanations = explanations[:, :, 1]
    explanations.feature_names = names.tolist()
    rows = []
    for i in range(3):
        shap.plots.waterfall(explanations[i], show=False, max_display=10)
        plt.savefig(root / f"figures/shap_waterfall_patient_{i+1}.png", dpi=170, bbox_inches="tight")
        plt.close("all")
        for j, name in enumerate(names):
            rows.append(dict(sample_patient=i+1, encoded_feature=name,
                transformed_value=float(encoded[i,j]), shap_value=float(explanations.values[i,j]),
                expected_value=float(explanations.base_values[i])))
    pd.DataFrame(rows).to_csv(root / "tables/shap_local_patients.csv", index=False)


def _calibrated_group_cv(pipeline, X, y, groups, folds, root):
    """Five-fold assessment with fold-local calibration and threshold selection.

    The chosen hyperparameters were selected with this training cohort, so these
    are post-selection CV estimates, not an unbiased nested-search estimate.
    Only the held-out test provides the final independent assessment.
    """
    rows, predictions = [], []
    for fold, (fit_idx, held_idx) in enumerate(folds, 1):
        x_outer, y_outer, g_outer = X.iloc[fit_idx], y.iloc[fit_idx], groups.iloc[fit_idx]
        base_idx, rest = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED).split(x_outer, groups=g_outer))
        cal_rel, val_rel = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED).split(rest, groups=g_outer.iloc[rest]))
        cal_idx, val_idx = rest[cal_rel], rest[val_rel]
        fitted = clone(pipeline).fit(x_outer.iloc[base_idx], y_outer.iloc[base_idx])
        calibrated = _calibrate(fitted, x_outer.iloc[cal_idx], y_outer.iloc[cal_idx])
        thresholds = choose_thresholds(y_outer.iloc[val_idx], calibrated.predict_proba(x_outer.iloc[val_idx])[:, 1])
        p = calibrated.predict_proba(X.iloc[held_idx])[:, 1]
        for name, threshold in thresholds.items():
            rows.append(dict(split="CV", fold=fold, operating_point=name, **compute_metrics(y.iloc[held_idx], p, threshold)))
        predictions.append(pd.DataFrame(dict(train_position=held_idx, fold=fold, y_true=y.iloc[held_idx].to_numpy(), probability=p)))
        print(f"  Final classifier grouped CV fold {fold}/5 complete", flush=True)
    pd.DataFrame(rows).to_csv(root / "tables/classifier_cv_folds.csv", index=False)
    pd.concat(predictions).to_csv(root / "tables/classifier_cv_predictions.csv", index=False)
    return rows


def run_classifier(frame, splits, output_dir=Path(".")):
    root = Path(output_dir)
    for path in ["tables", "artifacts", "figures"]:
        (root / path).mkdir(exist_ok=True, parents=True)
    cols = feature_columns(frame)
    forbidden = {"readmitted", "readmit_30", "patient_nbr", "encounter_id", "weight"}
    assert not forbidden.intersection(cols)
    X = {name: frame.iloc[index][cols].reset_index(drop=True) for name, index in splits.items()}
    y = {name: frame.iloc[index].readmit_30.reset_index(drop=True) for name, index in splits.items()}
    groups = frame.iloc[splits["train"]].patient_nbr.reset_index(drop=True)
    folds = grouped_cv(X["train"], y["train"], groups)
    ratio = float((1-y["train"].mean())/y["train"].mean())
    estimators = {
        "Logistic Regression": LogisticRegression(C=.1, class_weight="balanced", max_iter=700, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=160, min_samples_leaf=12, max_features=.6, class_weight="balanced_subsample", n_jobs=2, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=220, max_depth=3, learning_rate=.045, min_child_weight=12, subsample=.85, colsample_bytree=.85, reg_lambda=8, scale_pos_weight=ratio, tree_method="hist", n_jobs=2, eval_metric="logloss", random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=220, num_leaves=15, learning_rate=.045, min_child_samples=70, colsample_bytree=.85, reg_lambda=8, scale_pos_weight=ratio, n_jobs=2, verbosity=-1, random_state=SEED, deterministic=True),
    }
    candidates, comparison = {}, []
    for name, estimator in estimators.items():
        pipeline = make_pipeline(estimator)
        summary = _cv_summary(pipeline, X["train"], y["train"], folds)
        start = time.perf_counter()
        pipeline.fit(X["train"], y["train"])
        candidates[name] = pipeline
        row = dict(model=name, **summary, full_fit_seconds=time.perf_counter()-start)
        comparison.append(row)
        print(f"  {name}: grouped CV PR-AUC = {summary['cv_pr_auc_mean']:.4f}", flush=True)
    # Tune both boosting families rather than optimizing majority-class accuracy.
    search_spaces = {
        "XGBoost": {"model__max_depth": [2,3,4], "model__min_child_weight": [8,20,40], "model__n_estimators": [220,350], "model__learning_rate": [.035,.06], "model__scale_pos_weight": [1.,3.,ratio], "model__reg_lambda": [8,20]},
        "LightGBM": {"model__num_leaves": [7,15,23], "model__min_child_samples": [50,100,180], "model__n_estimators": [220,350], "model__learning_rate": [.035,.06], "model__scale_pos_weight": [1.,3.,ratio], "model__reg_lambda": [8,20]},
    }
    search_records = []
    for name, params in search_spaces.items():
        search = RandomizedSearchCV(make_pipeline(estimators[name]), params, n_iter=5,
            scoring="average_precision", cv=folds, n_jobs=1, random_state=SEED, refit=True, error_score="raise")
        search.fit(X["train"], y["train"])
        tuned_name = name + " (tuned)"
        candidates[tuned_name] = search.best_estimator_
        best_idx = search.best_index_
        comparison.append(dict(model=tuned_name, cv_pr_auc_mean=float(search.best_score_),
            cv_pr_auc_std=float(search.cv_results_["std_test_score"][best_idx]), full_fit_seconds=float(search.refit_time_)))
        pd.DataFrame(search.cv_results_).to_csv(root / f"tables/{name.lower()}_search.csv", index=False)
        search_records.append(dict(model=name, params=search.best_params_, cv_pr_auc=float(search.best_score_)))
        print(f"  Tuned {name}: grouped CV PR-AUC = {search.best_score_:.4f}", flush=True)
    candidate_table = pd.DataFrame(comparison).sort_values("cv_pr_auc_mean", ascending=False)
    chosen_name = candidate_table.iloc[0].model
    # Tree explanations are central to the proposal. If a linear baseline wins,
    # still use its true result in comparison and explicitly choose best tree.
    tree_names = candidate_table[candidate_table.model != "Logistic Regression"]
    chosen_name = tree_names.iloc[0].model
    base = candidates[chosen_name]
    candidate_table.to_csv(root / "tables/model_comparison.csv", index=False)
    # Same estimator and same folds isolate the resampling strategy comparison.
    imbalance_rows = []
    strategies = {
        "Class weighting": (ratio, None),
        "SMOTE": (1., SMOTE(sampling_strategy=.4, random_state=SEED)),
        "BorderlineSMOTE": (1., BorderlineSMOTE(sampling_strategy=.4, random_state=SEED)),
        "Random undersampling": (1., RandomUnderSampler(sampling_strategy=.5, random_state=SEED)),
        "Natural training + threshold tuning": (1., None),
    }
    for name, (weight, sampler) in strategies.items():
        classifier = LGBMClassifier(n_estimators=140, num_leaves=15, learning_rate=.05, min_child_samples=100,
            scale_pos_weight=weight, n_jobs=2, random_state=SEED, verbosity=-1, deterministic=True)
        pipe = make_pipeline(classifier, sampler)
        summary = _cv_summary(pipe, X["train"], y["train"], folds)
        pipe.fit(X["train"], y["train"])
        calibrated = _calibrate(pipe, X["calibration"], y["calibration"])
        val_p = calibrated.predict_proba(X["validation"])[:, 1]
        threshold = choose_thresholds(y["validation"], val_p)["balanced"]
        val_metrics = compute_metrics(y["validation"], val_p, threshold)
        imbalance_rows.append(dict(strategy=name, estimator="LightGBM (fixed settings)", **summary,
            validation_threshold=threshold, validation_accuracy=val_metrics["accuracy"], validation_recall=val_metrics["recall"],
            validation_pr_auc=val_metrics["pr_auc"], validation_brier=val_metrics["brier"]))
        print(f"  Imbalance experiment {name}: CV PR-AUC {summary['cv_pr_auc_mean']:.4f}", flush=True)
    pd.DataFrame(imbalance_rows).to_csv(root / "tables/imbalance_comparison.csv", index=False)
    selected, importance = explain_and_select(base, candidates.get("XGBoost (tuned)", candidates["XGBoost"]), X["train"], X["validation"], y["validation"], root)
    reduced = clone(base)
    start = time.perf_counter()
    reduced.fit(X["train"][selected], y["train"])
    reduced_seconds = time.perf_counter()-start
    all_seconds = float(candidate_table.set_index("model").loc[chosen_name, "full_fit_seconds"])
    fit_variants = {"All features": (base, cols, all_seconds), "Reduced features": (reduced, selected, reduced_seconds)}
    calibrated_variants, thresholds_by_variant, feature_rows = {}, {}, []
    for label, (pipeline, features, seconds) in fit_variants.items():
        calibrated = _calibrate(pipeline, X["calibration"][features], y["calibration"])
        calibrated_variants[label] = calibrated
        val_p = calibrated.predict_proba(X["validation"][features])[:, 1]
        thresholds_by_variant[label] = choose_thresholds(y["validation"], val_p)
        feature_rows.append(dict(feature_set=label, split="validation", training_seconds=seconds,
            features=len(pipeline.named_steps["drop"].kept_columns_),
            **compute_metrics(y["validation"], val_p, thresholds_by_variant[label]["balanced"])))
    # Prefer simpler model only if validation AP is within .005 of the full model.
    val_results = pd.DataFrame(feature_rows).set_index("feature_set")
    recommended = "Reduced features" if val_results.loc["Reduced features", "pr_auc"] >= val_results.loc["All features", "pr_auc"]-.005 else "All features"
    final_base, final_cols, _ = fit_variants[recommended]
    final_model = calibrated_variants[recommended]
    thresholds = thresholds_by_variant[recommended]
    save_json(root / "artifacts/selection_lock.json", dict(selected_model=chosen_name, feature_set=recommended,
        feature_columns=final_cols, thresholds=thresholds, selection_metric="grouped CV average precision; validation AP for reduced variant",
        validation_accuracy_floor=.82, validation_recall_goal=.70,
        locked_before_test_evaluation=True, randomized_search=search_records,
        imbalance_note="Resampling is a controlled comparison; final model comes from the tuned natural/class-weighted estimator family."))
    cv_rows = _calibrated_group_cv(final_base, X["train"][final_cols], y["train"], groups, folds, root)
    # First evaluation of held-out labels for any model selection is now over.
    metric_rows = []
    for split in ["train", "calibration", "validation", "test"]:
        p = final_model.predict_proba(X[split][final_cols])[:, 1]
        for operating_point, threshold in thresholds.items():
            metric_rows.append(dict(split=split, operating_point=operating_point, **compute_metrics(y[split], p, threshold)))
    cv_frame = pd.DataFrame(cv_rows)
    metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "threshold", "tn", "fp", "fn", "tp", "n", "prevalence", "selection_rate"]
    for name, subset in cv_frame.groupby("operating_point"):
        metric_rows.append(dict(split="CV mean", operating_point=name, **{col: float(subset[col].mean()) for col in metric_cols}))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(root / "tables/classifier_metrics.csv", index=False)
    for label, (pipeline, features, seconds) in fit_variants.items():
        p = calibrated_variants[label].predict_proba(X["test"][features])[:, 1]
        feature_rows.append(dict(feature_set=label, split="test", training_seconds=seconds,
            features=len(pipeline.named_steps["drop"].kept_columns_),
            **compute_metrics(y["test"], p, thresholds_by_variant[label]["balanced"])))
    pd.DataFrame(feature_rows).to_csv(root / "tables/feature_comparison.csv", index=False)
    test_p = final_model.predict_proba(X["test"][final_cols])[:, 1]
    test_frame = frame.iloc[splits["test"]].reset_index(drop=True)
    predictions = test_frame[["patient_nbr", "encounter_id", "race", "gender", "age"]].copy()
    predictions["y_true"] = y["test"]
    predictions["probability"] = test_p
    for name, threshold in thresholds.items():
        predictions["prediction_"+name] = (test_p >= threshold).astype(int)
    predictions.to_csv(root / "tables/test_predictions.csv", index=False)
    deciles = save_diagnostic_plots(y["test"], test_p, thresholds, root, "classifier")
    threshold_curve(y["validation"], final_model.predict_proba(X["validation"][final_cols])[:, 1]).to_csv(root / "tables/validation_threshold_curve.csv", index=False)
    fairness = fairness_table(test_frame, test_p, thresholds)
    fairness.to_csv(root / "tables/fairness.csv", index=False)
    _plot_fairness(fairness, root)
    # Train-only defaults allow dashboard inputs to omit less common measurements.
    defaults = {}
    for c in cols:
        defaults[c] = float(X["train"][c].median()) if pd.api.types.is_numeric_dtype(X["train"][c]) else str(X["train"][c].mode().iloc[0])
    save_json(root / "artifacts/feature_defaults.json", defaults)
    joblib.dump(final_model, root / "artifacts/classifier.joblib", compress=3)
    joblib.dump(final_base, root / "artifacts/base_pipeline.joblib", compress=3)
    # All-feature explainer remains for selection plots; dashboard explains the deployed estimator.
    import shap
    joblib.dump(shap.TreeExplainer(final_base.named_steps["model"]), root / "artifacts/shap_explainer.joblib", compress=3)
    save_deployed_explanations(final_base, X["train"][final_cols], root)
    summary = dict(selected_model=chosen_name, recommended_features=recommended, thresholds=thresholds,
        feature_columns=final_cols, metrics=metrics.to_dict("records"), top20_capture=float(deciles.iloc[1].cumulative_capture),
        majority_baseline_accuracy=float(1-y["test"].mean()), test_prevalence=float(y["test"].mean()),
        accuracy_requirement_met=bool(metrics.query("split == 'test' and operating_point == 'balanced'").accuracy.iloc[0] >= .8),
        high_recall_requirement_met=bool(metrics.query("split == 'test' and operating_point == 'high_recall'").recall.iloc[0] >= .6),
        cv_note="Five patient-grouped folds with fold-local calibration/thresholds. Hyperparameters selected on this cohort; post-selection CV is optimistic relative to nested search.",
        feature_selection_note="Reduced features chosen using training-only SHAP. Feature list is fixed before post-selection CV; feature selection is not nested, so independent test is the primary estimate.",
        explanation_units="SHAP explains raw model output; sigmoid calibration is separate.",
        sampler_note="SMOTE/BorderlineSMOTE interpolate one-hot encodings; fractional dummy values are an experimental limitation. Validation/test are never resampled.")
    save_json(root / "artifacts/classifier_metrics.json", summary)
    save_json(root / "artifacts/model_metadata.json", dict(selected_model=chosen_name, feature_set=recommended,
        feature_columns=final_cols, thresholds=thresholds, expected_accuracy_floor=.8,
        score_time="At discharge", explanation_units="uncalibrated base model raw output", random_seed=SEED))
    feature_summary = dict(recommended=recommended, selection_rule="Reduced if validation PR-AUC within .005 of all features; otherwise all features.",
        selected_features=selected, comparison=feature_rows, importance=importance.head(10).to_dict("records"))
    save_json(root / "artifacts/feature_selection.json", feature_summary)
    save_json(root / "artifacts/fairness.json", dict(scope="Held-out patients; descriptive encounter-level rates. Wilson recall intervals are approximate because repeated encounters are correlated; no causal or clinical fairness claim.",
        metrics=json.loads(fairness.to_json(orient="records"))))
    print("Phase 3/4 classifier completed:\n"+metrics.query("split == 'test'")[["operating_point", "accuracy", "recall", "precision", "pr_auc", "roc_auc"]].to_string(index=False), flush=True)
    return summary


def _plot_fairness(table, root):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, attr in zip(axes, ["race", "gender", "age"]):
        part = table[(table.attribute == attr) & (table.operating_point == "balanced")]
        positions = np.arange(len(part))
        ax.bar(positions-.18, part.recall, width=.36, label="Recall", color="#148f87")
        ax.bar(positions+.18, part.selection_rate, width=.36, label="Selection rate", color="#94d7d0")
        ax.set_xticks(positions, part.group, rotation=40, ha="right", fontsize=8)
        ax.set(title=attr.title(), ylim=(0, 1))
        sns.despine(ax=ax)
    axes[0].legend()
    fig.suptitle("Held-out fairness audit — balanced operating point; see CSV for denominators / intervals")
    fig.tight_layout()
    fig.savefig(root / "figures/fairness.png", dpi=170)
    plt.close(fig)
