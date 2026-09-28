"""Streamlit integration tests against the actual generated project artifacts."""
from pathlib import Path
import unittest

import joblib
import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
DATA_READY = (ROOT / "data/cleaned_data.csv").exists()
MODEL_READY = all((ROOT / path).exists() for path in (
    "artifacts/model_metadata.json", "artifacts/classifier.joblib",
    "artifacts/feature_defaults.json", "tables/test_predictions.csv"))


def _widget_by_label(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


class DashboardTests(unittest.TestCase):
    @unittest.skipUnless(DATA_READY, "Generate cleaned data before testing the dashboard.")
    def test_four_tabs_and_eda_filter(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 4)
        before = int(_widget_by_label(app.metric, "Eligible encounters").value.replace(",", ""))
        app.multiselect(key="filter_diag_1_group").set_value(["Diabetes"]).run()
        self.assertEqual(len(app.exception), 0)
        after = int(_widget_by_label(app.metric, "Eligible encounters").value.replace(",", ""))
        self.assertGreater(after, 0)
        self.assertLess(after, before)

    @unittest.skipUnless(MODEL_READY, "Train and save the classifier before testing inference.")
    def test_score_recomputes_features_and_matches_saved_model(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        for label, value in (("Prior inpatient visits", 3), ("Prior emergency visits", 1),
                             ("Prior outpatient visits", 2), ("Hospital stay (days)", 10),
                             ("Laboratory procedures", 50), ("Number of medications", 17)):
            _widget_by_label(app.number_input, label).set_value(value)
        app.selectbox(key="score_insulin").set_value("Up")
        app.selectbox(key="score_admission").set_value("Unknown")
        _widget_by_label(app.button, "Calculate readmission risk").click().run()
        self.assertEqual(len(app.exception), 0)
        scored = app.session_state["risk_prediction"]
        full = scored["cluster_row"].iloc[0]
        self.assertEqual(full.total_prior_visits, 6)
        self.assertEqual(full.lab_procs_per_day, 5)
        self.assertEqual(full.polypharmacy, 1)
        self.assertGreaterEqual(full.med_changes_count, 1)
        self.assertEqual(full.change, "Ch")
        probability = scored["probability"]
        self.assertTrue(np.isfinite(probability) and 0 <= probability <= 1)
        expected = joblib.load(ROOT / "artifacts/classifier.joblib").predict_proba(scored["row"])[0, 1]
        self.assertAlmostEqual(probability, expected, places=10)

    @unittest.skipUnless(MODEL_READY, "Generate held-out predictions before budget integration tests.")
    def test_budget_zero_and_unaffordable_scenarios(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        budget = _widget_by_label(app.slider, "Total follow-up budget ($)")
        budget.set_value(0)
        _widget_by_label(app.button, "Optimize follow-up plan").click().run()
        self.assertEqual(len(app.exception), 0)
        summary = app.session_state["budget_result"][0]["summary"]
        self.assertEqual(summary["patients_selected"], 0)
        self.assertEqual(summary["spending"], 0)
        self.assertIsNone(summary["roi"])
        _widget_by_label(app.slider, "Total follow-up budget ($)").set_value(50)
        app.slider(key="cost_0").set_value(70)
        _widget_by_label(app.button, "Optimize follow-up plan").click().run()
        self.assertEqual(len(app.exception), 0)
        summary = app.session_state["budget_result"][0]["summary"]
        self.assertEqual(summary["patients_selected"], 0)
        self.assertTrue(summary["budget_feasible"])


if __name__ == "__main__":
    unittest.main()
