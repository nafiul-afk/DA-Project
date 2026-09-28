"""Focused checks for leakage boundaries and portable recurrent artifacts."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from readmitrisk.sequences import (
    _feature_frame, _fit_prepare, _metric_suite, _predict_recurrent, _tensorflow,
    _thresholds, _train_recurrent, make_history_index,
)


class SequenceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "encounter_id": np.arange(1, 21),
            "patient_nbr": np.repeat([10, 20, 30, 40], 5),
            "readmit_30": np.tile([0, 1, 0, 0, 1], 4),
            "number_inpatient": np.arange(20, dtype=float),
            "age": ["[40-50)"] * 20,
            "age_ordinal": np.repeat([45, 55, 65, 75], 5),
            "gender": np.tile(["Female", "Male", "Female", "Male", "Female"], 4),
        })
        self.splits = {
            name: np.arange(5 * position, 5 * position + 5)
            for position, name in enumerate(["train", "calibration", "validation", "test"])
        }

    def test_histories_are_strictly_prior_and_padding_is_masked(self):
        histories, targets, lengths, splits = make_history_index(self.frame, self.splits)
        np.testing.assert_array_equal(histories[0], [-1, -1, 0])
        np.testing.assert_array_equal(histories[3], [1, 2, 3])
        for history, target in zip(histories, targets):
            actual = history[history >= 0]
            self.assertTrue(np.all(self.frame.iloc[actual].encounter_id < self.frame.iloc[target].encounter_id))
            self.assertTrue(np.all(self.frame.iloc[actual].patient_nbr == self.frame.iloc[target].patient_nbr))
        features, numeric, categorical = _feature_frame(self.frame)
        _, x = _fit_prepare(features, numeric, categorical, histories, splits["train"], np.arange(len(targets)))
        self.assertGreater(x.shape[2], 0)
        self.assertTrue(np.all(x[0, :2, :] == 0))
        self.assertEqual(lengths.tolist(), [1, 2, 3, 3] * 4)

    def test_preparation_learns_only_training_history_rows(self):
        histories, targets, _, splits = make_history_index(self.frame, self.splits)
        features, numeric, categorical = _feature_frame(self.frame)
        preparation, x = _fit_prepare(features, numeric, categorical, histories, splits["train"], np.arange(len(targets)))
        scaler = preparation.named_steps["encode"].named_transformers_["numeric"].named_steps["scale"]
        # Target-only fifth train encounter and all held-out patients are excluded.
        np.testing.assert_allclose(scaler.mean_, [1.5])
        changed = self.frame.copy()
        changed.loc[4, "number_inpatient"] = 1e9
        changed["readmit_30"] = 1 - changed.readmit_30
        changed_features, numeric, categorical = _feature_frame(changed)
        self.assertNotIn("readmit_30", changed_features.columns)
        self.assertNotIn("patient_nbr", changed_features.columns)
        self.assertNotIn("encounter_id", changed_features.columns)
        self.assertNotIn("age", changed_features.columns)
        _, changed_x = _fit_prepare(changed_features, numeric, categorical, histories, splits["train"], np.arange(len(targets)))
        np.testing.assert_array_equal(x, changed_x)

    def test_patient_overlap_fails_loudly(self):
        broken = {name: indices.copy() for name, indices in self.splits.items()}
        broken["train"] = np.array([0, 1, 2, 3, 5])
        broken["calibration"] = np.array([4, 6, 7, 8, 9])
        with self.assertRaisesRegex(ValueError, "patient"):
            make_history_index(self.frame, broken)

    def test_operating_thresholds_are_reproducible_on_given_validation_data(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        p = np.array([0.02, 0.05, 0.10, 0.60, 0.40, 0.90])
        thresholds = _thresholds(y, p, minimum_accuracy=0.8, target_recall=0.7)
        self.assertAlmostEqual(thresholds["balanced"], 0.4)
        self.assertAlmostEqual(thresholds["high_recall"], 0.4)
        metrics = _metric_suite(y, p, thresholds["balanced"])
        self.assertAlmostEqual(metrics["accuracy"], 5 / 6)
        self.assertEqual(metrics["recall"], 1.0)

    def test_gru_lstm_train_and_saved_model_predictions_agree(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(48, 3, 8)).astype(np.float32)
        x[::2, :2, :] = 0
        y = np.tile([0, 0, 0, 1], 12).astype(np.int32)
        for name in ["GRU", "LSTM"]:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temporary:
                model, history = _train_recurrent(name, x[:32], y[:32], x[32:], y[32:], epochs=1, seed=42)
                self.assertEqual(len(history["loss"]), 1)
                before = _predict_recurrent(model, x[32:])
                path = Path(temporary) / f"{name.lower()}.keras"
                model.save(path)
                restored = _tensorflow().keras.models.load_model(path)
                after = _predict_recurrent(restored, x[32:])
                np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
                self.assertTrue(np.isfinite(before).all())


if __name__ == "__main__":
    unittest.main()
