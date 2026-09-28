"""Checks for leakage boundaries and threshold behavior, not copied outputs."""
import numpy as np
import pandas as pd
import pytest
from readmitrisk.data import (NearZeroDropper, RareCategoryGrouper, diagnosis_group,
                             feature_columns, patient_splits, read_id_mappings)
from readmitrisk.evaluation import choose_thresholds, compute_metrics


@pytest.mark.parametrize("code,expected", [
    ("250.83", "Diabetes"), ("249", "Other"), ("459.9", "Circulatory"),
    ("460", "Respiratory"), ("519.9", "Respiratory"), ("520", "Digestive"),
    ("785.4", "Circulatory"), ("786.5", "Respiratory"), ("787.2", "Digestive"),
    ("788.2", "Genitourinary"), ("710", "Musculoskeletal"), ("999", "Injury"),
    ("239.9", "Neoplasms"), ("E885", "Other"), ("V45", "Other"),
    (np.nan, "Other"), ("?", "Other"),
])
def test_icd_boundaries(code, expected):
    assert diagnosis_group(code) == expected


def test_patient_splits_keep_repeated_encounters_together():
    frame = pd.DataFrame(dict(patient_nbr=np.repeat(np.arange(200), np.arange(200)%5+1)))
    splits = patient_splits(frame)
    sets = [set(frame.iloc[idx].patient_nbr) for idx in splits.values()]
    assert len(set.union(*sets)) == 200
    assert sum(map(len, sets)) == 200
    assert np.array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(frame)))


def test_preprocessing_does_not_learn_unseen_validation_values():
    train = pd.DataFrame({"medical_specialty": ["Common"]*98+["Rare", "Unknown"],
                          "constant_drug": ["No"]*100, "numeric": range(100)})
    valid = pd.DataFrame({"medical_specialty": ["Unseen"], "constant_drug": ["Up"], "numeric": [1000000]})
    dropper = NearZeroDropper().fit(train)
    grouper = RareCategoryGrouper(min_frequency=.02).fit(dropper.transform(train))
    transformed = grouper.transform(dropper.transform(valid))
    assert "constant_drug" not in transformed
    assert transformed.medical_specialty.iloc[0] == "Other"
    assert "Unseen" not in grouper.levels_["medical_specialty"]


def test_identifiers_and_outcome_never_model_features():
    frame = pd.DataFrame(columns=["patient_nbr", "encounter_id", "readmitted", "readmit_30", "age", "age_ordinal", "time_in_hospital"])
    assert feature_columns(frame) == ["age_ordinal", "time_in_hospital"]


def test_thresholds_select_best_validation_recall_with_accuracy_margin():
    rng = np.random.default_rng(42)
    y = np.r_[np.ones(100), np.zeros(900)]
    p = np.clip(.1+.2*y+rng.normal(0, .09, len(y)), 0, 1)
    thresholds = choose_thresholds(y, p)
    balanced = compute_metrics(y, p, thresholds["balanced"])
    high = compute_metrics(y, p, thresholds["high_recall"])
    assert balanced["accuracy"] >= .82
    assert high["recall"] >= .70
    feasible_recalls = [compute_metrics(y, p, t)["recall"] for t in np.unique(p) if compute_metrics(y, p, t)["accuracy"] >= .82]
    assert balanced["recall"] == max(feasible_recalls)


def test_uci_mapping_has_real_sections():
    mappings = read_id_mappings("data/IDS_mapping.csv")
    assert mappings["admission_type_id"][1] == "Emergency"
    assert "Expired" in mappings["discharge_disposition_id"][11]
    assert mappings["admission_source_id"][7] == "Emergency Room"
