"""Deterministic clinical features and train-only learned preprocessing.

Identifiers stay in the analysis frame for splitting and sequence ordering;
``feature_columns`` prevents either identifier entering a predictive model.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
EXCLUDED_DISCHARGES = {11, 13, 14, 19, 20, 21}
MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
EXCLUDE_FEATURES = {
    "patient_nbr", "encounter_id", "readmitted", "readmit_30", "weight", "age",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "diag_1", "diag_2", "diag_3", "split",
}


def save_json(path, value):
    """Save numpy/pandas scalar values as portable, strict JSON."""
    def convert(x):
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(f"Cannot serialize {type(x)}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False))


def read_id_mappings(path):
    """UCI's file contains three CSV tables separated by blank rows."""
    mappings, current = {}, None
    with open(path, newline="", encoding="utf-8-sig") as file:
        for row in csv.reader(file):
            if not row or not row[0].strip():
                continue
            key = row[0].strip()
            if key.endswith("_id"):
                current = key
                mappings[current] = {}
            elif current and key.isdigit():
                description = ",".join(row[1:]).strip()
                mappings[current][int(key)] = description or "Unknown"
    expected = {"admission_type_id", "discharge_disposition_id", "admission_source_id"}
    if set(mappings) != expected:
        raise ValueError(f"Incomplete ID mapping: {set(mappings)}")
    return mappings


def diagnosis_group(value):
    """Map ICD-9 codes by the proposal's mutually exclusive definitions."""
    text = str(value).strip().upper()
    if text.startswith(("V", "E")):
        return "Other"
    try:
        code = float(text)
    except (ValueError, TypeError):
        return "Other"
    if 250 <= code < 251:
        return "Diabetes"
    for name, lower, upper, extra in [
        ("Circulatory", 390, 460, 785), ("Respiratory", 460, 520, 786),
        ("Digestive", 520, 580, 787), ("Injury", 800, 1000, None),
        ("Musculoskeletal", 710, 740, None), ("Genitourinary", 580, 630, 788),
        ("Neoplasms", 140, 240, None),
    ]:
        if lower <= code < upper or (extra is not None and extra <= code < extra + 1):
            return name
    return "Other"


def _readable_group(description):
    text = str(description).strip()
    if text.lower() in {"null", "not available", "not mapped", "unknown/invalid", "unknown", "nan"}:
        return "Unknown"
    return text


def engineer_features(raw, mappings):
    """Only row-wise, prespecified operations: safe before the patient split."""
    frame = raw.copy().replace("?", np.nan)
    frame = frame.loc[~frame.discharge_disposition_id.isin(EXCLUDED_DISCHARGES)].copy()
    frame = frame.drop_duplicates("encounter_id", keep="first")
    frame["readmit_30"] = frame.readmitted.eq("<30").astype("int8")
    for column in ["race", "gender", "payer_code", "medical_specialty"]:
        frame[column] = frame[column].fillna("Unknown").replace("Unknown/Invalid", "Unknown")
    for column in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        frame[column.removesuffix("_id")] = frame[column].map(mappings[column]).fillna("Unknown").map(_readable_group)
    for column in ["diag_1", "diag_2", "diag_3"]:
        frame[column + "_group"] = frame[column].map(diagnosis_group)
    frame["age_ordinal"] = frame.age.str.extract(r"\[(\d+)-")[0].astype(float) + 5
    frame["total_prior_visits"] = frame[["number_outpatient", "number_emergency", "number_inpatient"]].sum(axis=1)
    frame["med_changes_count"] = frame[MEDICATIONS].isin(["Up", "Down"]).sum(axis=1)
    frame["meds_active_count"] = (frame[MEDICATIONS].notna() & frame[MEDICATIONS].ne("No")).sum(axis=1)
    frame["lab_procs_per_day"] = frame.num_lab_procedures / frame.time_in_hospital.clip(lower=1)
    frame["diag_diversity"] = frame[["diag_1_group", "diag_2_group", "diag_3_group"]].nunique(axis=1)
    frame["polypharmacy"] = frame.num_medications.gt(15).astype("int8")
    # 'None' is a measured absence of testing, not an accidental pandas NA.
    for column in ["A1Cresult", "max_glu_serum"]:
        frame[column] = frame[column].replace({"None": "Not tested"}).fillna("Not tested")
    return frame.drop(columns=["weight", "readmitted", "diag_1", "diag_2", "diag_3"]).reset_index(drop=True)


def feature_columns(frame):
    return [column for column in frame if column not in EXCLUDE_FEATURES]


def patient_splits(frame):
    """80/20 development/test; development divided 60/10/10 overall.

    Model fitting and five-fold CV use train; calibration uses its own patients;
    validation chooses thresholds. Test labels have no role in these decisions.
    GroupShuffleSplit splits *patients*, so encounter fractions are approximate.
    """
    ids = np.arange(len(frame))
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(ids, groups=frame.patient_nbr))
    tr, hold = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED).split(dev, groups=frame.iloc[dev].patient_nbr))
    train, holdout = dev[tr], dev[hold]
    ca, va = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED).split(holdout, groups=frame.iloc[holdout].patient_nbr))
    splits = dict(train=train, calibration=holdout[ca], validation=holdout[va], test=test)
    names = list(splits)
    for i, name in enumerate(names):
        for other in names[i+1:]:
            assert set(frame.iloc[splits[name]].patient_nbr).isdisjoint(frame.iloc[splits[other]].patient_nbr)
    assert sorted(np.concatenate(list(splits.values())).tolist()) == list(range(len(frame)))
    return splits


class NearZeroDropper(TransformerMixin, BaseEstimator):
    """Learn near-constant columns inside each training fold, never globally."""
    def __init__(self, threshold=.995):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.dropped_columns_ = [c for c in X if X[c].value_counts(normalize=True, dropna=False).iloc[0] > self.threshold]
        self.kept_columns_ = [c for c in X if c not in self.dropped_columns_]
        return self

    def transform(self, X):
        return X.loc[:, self.kept_columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_columns_, dtype=object)


class RareCategoryGrouper(TransformerMixin, BaseEstimator):
    """Unknown is retained; rare/unseen specialty and administrative levels pool."""
    def __init__(self, min_frequency=.01):
        self.min_frequency = min_frequency

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        pool = ["medical_specialty", "payer_code", "admission_type", "discharge_disposition", "admission_source"]
        self.levels_ = {}
        for c in pool:
            if c in X:
                frequencies = X[c].fillna("Unknown").value_counts(normalize=True)
                self.levels_[c] = set(frequencies[frequencies >= self.min_frequency].index) | {"Unknown"}
        return self

    def transform(self, X):
        result = X.copy()
        for column, levels in self.levels_.items():
            values = result[column].fillna("Unknown")
            result[column] = values.where(values.isin(levels), "Other")
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def make_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ], sparse_threshold=0)


def prepare_data(output_dir=Path(".")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(output_dir)
    for directory in ["data", "artifacts", "figures", "tables"]:
        (root / directory).mkdir(exist_ok=True, parents=True)
    path = root / "data/diabetic_data.csv"
    if not path.exists():
        raise FileNotFoundError("Place diabetic_data.csv and IDS_mapping.csv in ./data/. See README.md for the UCI download.")
    raw = pd.read_csv(path, keep_default_na=False, na_values=["?"], low_memory=False)
    assert len(raw) >= 2000 and raw.shape[1] - 3 >= 25, "Grading dataset minimum not met"
    mappings = read_id_mappings(root / "data/IDS_mapping.csv")
    missing = raw.isna().mean().sort_values(ascending=False).rename("missing_fraction").to_frame()
    missing["missing_count"] = raw.isna().sum()
    missing.to_csv(root / "tables/missingness.csv", index_label="feature")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    observed = missing.loc[missing.missing_fraction > 0].iloc[::-1]
    ax.barh(observed.index, observed.missing_fraction * 100, color="#148f87")
    for i, value in enumerate(observed.missing_fraction * 100):
        ax.text(value + .6, i, f"{value:.1f}%", va="center")
    ax.set(xlabel="Missing values (%)", xlim=(0, 106), title="Missingness before cleaning — all raw encounters")
    fig.text(.05, .01, f"Weight is {missing.loc['weight', 'missing_fraction']:.1%} missing; payer and specialty retain an Unknown category.", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(root / "figures/01_missingness.png", dpi=170)
    plt.close(fig)
    frame = engineer_features(raw, mappings)
    splits = patient_splits(frame)
    split_rows = []
    manifest = frame[["encounter_id", "patient_nbr"]].copy()
    for name, positions in splits.items():
        subset = frame.iloc[positions]
        manifest.loc[positions, "split"] = name
        split_rows.append(dict(split=name, encounters=len(subset), patients=subset.patient_nbr.nunique(), positives=int(subset.readmit_30.sum()), prevalence=float(subset.readmit_30.mean())))
    pd.DataFrame(split_rows).to_csv(root / "tables/split_summary.csv", index=False)
    manifest.to_csv(root / "artifacts/split_manifest.csv", index=False)
    np.savez_compressed(root / "artifacts/split_indices.npz", **splits)
    frame.to_csv(root / "data/cleaned_data.csv", index=False)
    dropped = NearZeroDropper().fit(frame.iloc[splits["train"]][feature_columns(frame)]).dropped_columns_
    quality = dict(raw_rows=len(raw), raw_columns=raw.shape[1], raw_predictors=raw.shape[1]-3,
        eligible_rows=len(frame), eligible_patients=frame.patient_nbr.nunique(),
        excluded_discharge_rows=int(raw.discharge_disposition_id.isin(EXCLUDED_DISCHARGES).sum()),
        duplicate_encounters_removed=int(raw.loc[~raw.discharge_disposition_id.isin(EXCLUDED_DISCHARGES)].encounter_id.duplicated().sum()),
        repeated_patient_encounters=int(frame.patient_nbr.duplicated().sum()),
        candidate_predictors=len(feature_columns(frame)), train_near_zero_columns=dropped,
        retained_predictors=len(feature_columns(frame))-len(dropped),
        eligible_prevalence=float(frame.readmit_30.mean()), patient_overlap=0,
        dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ids_mapping_sha256=hashlib.sha256((root / "data/IDS_mapping.csv").read_bytes()).hexdigest(),
        split_summary=split_rows,
        prediction_time="At discharge; LOS, final diagnosis and discharge plan must be available.",
        policy="No identifiers or outcome-derived variables in model inputs. Near-zero and rare-level rules fit inside folds.")
    save_json(root / "artifacts/data_quality.json", quality)
    print(f"Phase 1: {len(raw):,} x {raw.shape[1]} raw; {len(frame):,} eligible encounters; {frame.patient_nbr.nunique():,} patients; patient overlap = 0.", flush=True)
    return frame, splits, quality
