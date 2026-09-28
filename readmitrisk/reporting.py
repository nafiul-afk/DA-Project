"""Build the final report and 14-slide presentation from saved, measured results.

Nothing in this module trains on test data or invents results. Missing required
artifacts raise an error, so an incomplete run cannot silently become a final
presentation with example numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

DARK = "153C47"
TEAL = "0F8B8D"
MINT = "78C9B5"
PALE = "EAF5F2"
INK = "223D47"
GRAY = "58747A"
WHITE = "FFFFFF"
ORANGE = "F1A45B"


def _load(root: Path, filename: str):
    path = root / "artifacts" / filename
    if not path.exists():
        raise FileNotFoundError(f"Complete the analysis before building the presentation: missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _table(root: Path, filename: str) -> pd.DataFrame:
    path = root / "tables" / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing measured results table: {path}")
    return pd.read_csv(path)


def _value(value, column=""):
    if value is None or pd.isna(value):
        return "Not estimable"
    if column in {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "readmission_rate", "selection_rate", "prevalence", "cv_recall", "cv_accuracy", "validation_accuracy", "validation_recall"}:
        return f"{float(value):.1%}"
    if column in {"pr_auc", "roc_auc", "brier", "cv_pr_auc_mean", "cv_pr_auc_std", "mean_abs_shap", "threshold", "silhouette", "davies_bouldin", "validation_pr_auc", "validation_roc_auc", "validation_brier", "validation_threshold"}:
        return f"{float(value):.3f}"
    if column in {"p_value", "holm_adjusted_p"}:
        return f"{float(value):.3g}"
    if column in {"spending", "expected_net_savings"}:
        return f"${float(value):,.0f}"
    if column == "roi":
        return f"{float(value):.1%}"
    if column in {"expected_prevented_readmissions", "training_seconds", "mean_age", "mean_los", "mean_prior_visits", "mean_medications"}:
        return f"{float(value):,.1f}"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.3f}"
    return str(value)


def _markdown(frame: pd.DataFrame, columns=None):
    if columns is not None:
        frame = frame.loc[:, columns]
    names = list(frame.columns)
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join(["---"] * len(names)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_value(row[c], c).replace("|", "/") for c in names) + " |")
    return "\n".join(lines)


def _test_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "split" in frame:
        return frame.loc[frame["split"].astype(str).str.lower().eq("test")].copy()
    return frame.copy()


def _sequence_test_rows(sequence: dict) -> pd.DataFrame:
    rows = []
    for model, details in sequence["models"].items():
        for operating_point, values in details["metrics"]["test"].items():
            rows.append({"model": model, "operating_point": operating_point, **values})
    return pd.DataFrame(rows)


def _allocation_note(budget: dict) -> str:
    top_risk = next(row for row in budget["comparison"] if row["strategy"].startswith("Top risk"))
    difference = budget["optimized"]["expected_prevented_readmissions"] - top_risk["expected_prevented_readmissions"]
    if abs(difference) < 1e-7:
        return "The top-risk cheapest-tier baseline ties the optimizer under these uniform cost/effect assumptions; optimization adds no measured benefit over that simpler rule in this default scenario."
    return f"Under these assumptions, optimization changes expected prevented readmissions by {difference:.2f} compared with top-risk cheapest-tier allocation."


def _fairness_summary(root: Path):
    paths = [root / "tables/fairness.csv", root / "tables/fairness_by_group.csv", root / "tables/fairness_audit.csv"]
    found = next((p for p in paths if p.exists()), None)
    if found is None:
        return [], None
    frame = pd.read_csv(found)
    point = frame.loc[frame.operating_point.eq("balanced")]
    summaries = []
    for attribute, rows in point.groupby("attribute"):
        reliable = rows[(rows["n"] >= 100) & (rows["positives"] >= 20)]
        if len(reliable) >= 2:
            summaries.append(f"{attribute}: recall {reliable.recall.min():.1%}–{reliable.recall.max():.1%}; selection {reliable.selection_rate.min():.1%}–{reliable.selection_rate.max():.1%} across adequately sized groups.")
    return summaries, frame


def build_final_report(output_dir: Path | str = Path(".")) -> Path:
    root = Path(output_dir)
    quality = _load(root, "data_quality.json")
    eda = _load(root, "eda.json")
    classifier = _load(root, "classifier_metrics.json")
    feature_selection = _load(root, "feature_selection.json")
    sequence = _load(root, "sequence_metrics.json")
    budget = _load(root, "budget_metrics.json")
    clusters = _load(root, "clustering.json")
    metrics = _test_rows(_table(root, "classifier_metrics.csv"))
    features = _test_rows(_table(root, "feature_comparison.csv"))
    comparison = _table(root, "model_comparison.csv")
    imbalance = _table(root, "imbalance_comparison.csv")
    seq_rows = _sequence_test_rows(sequence)
    seq_test = next(row for row in sequence["sample_counts"] if row["split"] == "test")
    balanced = metrics.loc[metrics.operating_point.eq("balanced")].iloc[0]
    high = metrics.loc[metrics.operating_point.eq("high_recall")].iloc[0]
    optimized = budget["optimized"]
    fairness_lines, fairness = _fairness_summary(root)
    rows = [
        "# ReadmitRisk — Final Analysis Report", "",
        "**Group 8 · Data Analytics Laboratory · Summer 2026 · United International University**", "",
        "Nafiul Islam (0152410070) · Onika Tahmim Subha (0152330134)", "",
        "## Business question and decision timing", "",
        "Which eligible discharge encounters are most likely to be followed by a recorded readmission within 30 days, and how can a limited follow-up budget be allocated? Agent 1 is a discharge-time model: final stay length, discharge destination and treatment summaries must already be known. Agent 2 predicts the next recorded encounter's label using earlier history. The optimizer estimates scenario value; it does not estimate intervention effects from this observational dataset.", "",
        "## Dataset, cleaning and leakage controls", "",
        f"The raw file has **{quality['raw_rows']:,} rows and {quality['raw_columns']} columns** ({quality['raw_predictors']} candidate raw predictors after separating identifiers and target), exceeding the 2,000-row/25-feature requirement. Removing the specified discharge categories and duplicate encounter IDs leaves **{quality['eligible_rows']:,} encounters from {quality['eligible_patients']:,} patients**. The eligible target rate is {quality['eligible_prevalence']:.2%}.", "",
        f"{quality['excluded_discharge_rows']:,} expired/hospice-coded records were excluded. Weight was dropped; payer and specialty missingness became Unknown; rare categories and near-constant columns are handled by transformers fitted inside each training fold. Admission identifiers use the supplied mapping; diagnoses use the nine prespecified ICD-9 groups. Engineered features describe prior visits, medication activity/change, age, daily lab intensity, diagnostic diversity and polypharmacy.", "",
        "Patients are isolated across training, calibration, validation and test. Five-fold grouped CV is restricted to training patients. Calibration uses a separate patient set; validation determines operating thresholds. EDA uses the training partition only. Identifiers and outcome-derived variables never enter Agent 1 predictors. Encoders, scalers, sampling and feature selection learn from training data only.", "",
        _markdown(pd.DataFrame(quality["split_summary"])), "",
        "## EDA: key business insights", "",
    ]
    rows.extend(f"- {insight}" for insight in eda["key_business_insights"])
    rows.extend(["", eda["inference_design"], "", _markdown(pd.DataFrame(eda["hypotheses"]), ["id", "test", "p_value", "holm_adjusted_p", "effect_size", "conclusion"]), "",
        "Chart rates describe encounters, including repeat encounters; tests use one earliest eligible encounter per patient. H1 combines an omnibus chi-square with a prespecified direction check; it does not prove a monotonic causal relation. H2 is a one-sided distribution test, not a test of equal means. H3's separate comparisons are adjusted with H1/H2 using Holm's procedure. Effect sizes matter even when large samples produce very small p-values.", "",
        "## Agent 1: classifier comparison and final operating points", "",
        f"Selected classifier: **{classifier['selected_model']}**. Model selection uses grouped development PR-AUC, not held-out accuracy. PR-AUC is computed as average precision. Brier score measures probability error; smaller is better. The baseline PR-AUC of an uninformative ranking equals the target prevalence.", "",
        classifier["cv_note"] + " " + classifier["feature_selection_note"], "",
        _markdown(comparison, [c for c in ["model", "cv_pr_auc_mean", "cv_pr_auc_std", "validation_pr_auc", "validation_roc_auc"] if c in comparison]), "",
        "Natural-distribution, patient-held-out results:", "",
        _markdown(metrics, ["operating_point", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "threshold"]), "",
        f"The balanced point {'meets' if balanced.accuracy >= .8 else 'does not meet'} the requested ≥80% test accuracy: **{balanced.accuracy:.2%} accuracy**, **{balanced.recall:.2%} recall**, **{balanced.precision:.2%} precision**. The high-recall point {'meets' if high.recall >= .6 else 'does not meet'} the requested ≥60% recall: **{high.recall:.2%} recall**, **{high.accuracy:.2%} accuracy**, **{high.precision:.2%} precision**. Thresholds are frozen from validation; test labels do not select the operating point.", "",
        "The balanced rule maximizes validation recall subject to a predeclared 82% validation-accuracy floor, leaving a two-percentage-point buffer for the requested 80% test target. This conservative rule is not claimed to maximize recall retrospectively at exactly 80% test accuracy. The high-recall rule seeks at least 70% validation recall and reports the recall actually achieved on new patients.", "",
        f"A high accuracy score does not mean most readmissions were detected. A no-readmission classifier would achieve {classifier['majority_baseline_accuracy']:.2%} test accuracy and zero recall. The two operating points make the service-capacity trade-off visible. Their measured recall and precision are more useful for planning follow-up than an accuracy target alone.", "",
        "Imbalance experiments:", "", _markdown(imbalance), "",
        "Sampling is inside training pipelines only. CV recall/accuracy in the strategy table use each estimator's native decision rule; validation columns apply separate sigmoid calibration and the common validation threshold-selection rule. Compare CV PR-AUC for ranking and validation operating metrics for the tuned decision, rather than confusing these two evaluations. SMOTE-style interpolation of one-hot encoded clinical categories produces synthetic numeric vectors rather than literal patient records; interpret this as a benchmark strategy, not simulated patient physiology. The calibrated final model and the deployment threshold are separate decisions.", "",
        "## Agent 2: sequence forecasting", "", _markdown(pd.DataFrame(sequence["sample_counts"])), "",
        _markdown(seq_rows, ["model", "operating_point", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]), "",
        "All three sequence models use the same sequence-eligible target encounters and patient splits. For a target encounter, only earlier encounters are supplied. The XGBoost history baseline uses historical summaries; it has no access to target-encounter features. Encounter ID is an ordering proxy, with no true timing intervals. Selection into the repeated-encounter subset changes both prevalence and difficulty, so its metrics are not directly comparable with Agent 1's all-encounter test results.", "",
        f"The sequence test prevalence is {seq_test['readmission_rate']:.2%}; predicting no readmission for every sequence would already be {1-seq_test['readmission_rate']:.2%} accurate. Thus the measured ≥80% accuracy is compatible with very low balanced-point recall. The validation-selected history model is {sequence['selected_by_validation_pr_auc']}; recurrent architectures do not automatically improve on a boosted historical-feature baseline.", "",
        "Sequence accuracy requirements at the balanced operating point:", "",
    ])
    for _, row in seq_rows.loc[seq_rows.operating_point.eq("balanced")].iterrows():
        rows.append(f"- {row.model}: {row.accuracy:.2%} accuracy and {row.recall:.2%} recall; ≥80% accuracy {'met' if row.accuracy >= .8 else 'not met'}. Thresholds were chosen using sequence validation data.")
    rows.extend(["", "## Feature selection and model comparison", "",
        _markdown(features, ["feature_set", "accuracy", "recall", "f1", "pr_auc", "roc_auc", "training_seconds", "features"]), "",
        f"**Recommendation: {feature_selection['recommended']}.** The predeclared rule is: {feature_selection['selection_rule']} This choice uses validation evidence; the held-out table above audits the frozen decision. A reduced model is preferred when its validation discrimination remains close enough because it needs fewer inputs and is easier to explain.", "",
        "SHAP rankings, permutation importance and boosting gain supply complementary evidence. Medication activity flags are checked for pairwise correlation and variance inflation; constant or exact-redundant flags cannot provide independent information. SHAP feature ranks and the reduced subset are learned from training data; permutation importance is a separate validation-set diagnostic. SHAP is an explanation of the fitted model, not a causal effect, and raw booster margin explanations are distinguished from the calibrated risk probability.", "",
        "## Agent 3: follow-up budget scenarios", "",
        f"The allocator uses **{budget['unique_test_patients']:,} distinct test patients**, retaining one eligible scored encounter per patient. Budget: **${budget['assumptions']['budget']:,.0f}**. Assumed intervention costs are {budget['assumptions']['costs']}; relative risk reductions are {budget['assumptions']['effectiveness']}. Readmission cost is assumed to be ${budget['assumptions']['readmission_cost']:,.0f}.", "",
        _markdown(pd.DataFrame(budget["comparison"]), ["strategy", "patients_selected", "spending", "budget_feasible", "expected_prevented_readmissions", "expected_net_savings", "roi"]), "",
        f"The optimized scenario selects **{optimized['patients_selected']:,} patients**, spends **${optimized['spending']:,.0f}**, estimates **{optimized['expected_prevented_readmissions']:.2f} prevented readmissions**, and estimates **${optimized['expected_net_savings']:,.0f} net savings**. ROI is {optimized['roi']:.1%}, calculated as (avoided readmission cost − intervention spending) / intervention spending. Solver status: {optimized['solution_status']}; budget feasible: {optimized['budget_feasible']}.", "",
        _allocation_note(budget), "",
        "These expected benefits are generated by the assumptions, not observed in the dataset. Effectiveness varies in practice; no clinical cost-effectiveness conclusion is established here. Blanket allocation is explicitly labeled if infeasible and must not be compared as an equal-budget plan in that case. Budget sensitivity evaluates the same modeled planning problem across budgets, using shorter solver limits. The saved sensitivity statuses distinguish proven optima from feasible heuristic fallbacks with unknown gaps, so the curve is feasible expected benefit, not a certified optimal frontier.", "",
        "## Agent 4: patient risk-tier clustering", "",
        f"Training-fitted K-Means selected **k={clusters['selected_k']}** using the documented internal-metric rule. GaussianMixture is an independent model-family sanity check. K-Means silhouette is {clusters['algorithm_comparison'][0]['silhouette']:.3f}; this low value indicates weak geometric separation, so these are exploratory workflow segments rather than distinct clinical phenotypes. PCA is used to display the feature space, not as proof that clinical risk falls into discrete classes.", "",
        _markdown(pd.DataFrame(clusters["algorithm_comparison"])), "",
        _markdown(pd.DataFrame(clusters["test_profiles"])), "",
        _markdown(pd.DataFrame(clusters["playbooks"])), "",
        "Cluster labels and care-team playbooks are operational descriptions. Test outcome rates profile the fitted clusters after assignment; they do not train the clusters or prove treatment suitability.", "",
        "## Fairness and limitations", "",
    ])
    rows.extend(f"- {line}" for line in fairness_lines)
    rows.extend(["", "Subgroup metrics include denominators and unstable small groups remain visible. Recall confidence intervals are approximate encounter-level descriptions when repeated encounters remain. Rate differences may reflect confounding or different prevalence; this audit cannot establish fairness or a causal explanation.", "",
        "- Historical coverage: US hospital encounters from 1999–2008 may differ substantially from present-day Bangladesh.",
        "- Outcome coverage and recording practices may miss readmissions elsewhere; eligibility and repeated-history selection limit generalization.",
        "- Encounter ID is only a proxy for chronological order; no true time interval is learned.",
        "- Administrative and clinical summaries support limited discrimination; the achieved ROC-AUC is reported directly. There is no proven universal 0.65–0.70 ceiling for this dataset.",
        "- A patient-grouped random split checks new-patient generalization, not future-calendar drift or generalization to unseen hospitals.",
        "- Costs/effectiveness are editable scenario assumptions, not a health-economic evaluation. External, prospective validation and local calibration are necessary before clinical use.", "",
        "## Dashboard and submission", "",
        "Run `streamlit run app.py` in the project environment. The four tabs provide EDA filtering, live calibrated risk scoring with local drivers, editable budget simulation, and cluster profiles/playbooks. The sidebar switches operating points. See `docs/CRITERIA_CHECKLIST.md` for criterion mapping and `docs/VIVA_QUESTIONS.md` for viva preparation.", "",
        f"The dashboard's post-evaluation EDA view describes all {quality['eligible_rows']:,} cleaned encounters. Model-development EDA uses only {eda['encounters']:,} training encounters; its hypothesis tests use one earliest eligible encounter for each of {eda['patients']:,} training patients. Dashboard exploration is not used to select or retune the frozen model, so the two EDA views have different denominators by design.", "",
        "Team allocation follows the approved proposal: Nafiul — classifier and clustering; Onika — sequence forecasting and allocation; both — data cleaning/EDA, SHAP/model review, dashboard integration, testing and presentation. This states the agreed responsibility plan, not an assertion about independently observed individual contributions.", "",
    ])
    report = root / "docs/FINAL_REPORT.md"
    report.parent.mkdir(exist_ok=True, parents=True)
    report.write_text("\n".join(rows), encoding="utf-8")
    return report


class _Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333)
        self.prs.slide_height = Inches(7.5)

    def rect(self, slide, x, y, w, h, fill, radius=False):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                      Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(fill)
        shape.line.fill.background()
        return shape

    def text(self, slide, x, y, w, h, text, size=18, color=INK, bold=False, align=None):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.015)
        tf.margin_top = tf.margin_bottom = Inches(0.005)
        for i, line in enumerate(str(text).split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.name = "Aptos"
            p.font.size = Pt(size)
            p.font.bold = bold
            p.font.color.rgb = RGBColor.from_string(color)
            p.space_after = Pt(5)
            if align is not None:
                p.alignment = align
        return box

    def slide(self, title, kicker="READMITRISK / GROUP 8", dark=False):
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor.from_string(DARK if dark else "F8FBFA")
        self.rect(slide, 0, 0, 13.333, .11, TEAL)
        self.text(slide, .55, .32, 12.2, .3, kicker, 10, MINT if dark else TEAL, True)
        title_size = 29 if len(title) < 66 else (26 if len(title) < 76 else 24)
        self.text(slide, .55, .8, 12.15, .8, title, title_size, WHITE if dark else DARK, True)
        self.rect(slide, .55, 1.62, .64, .06, MINT)
        self.text(slide, .55, 7.15, 11.7, .2, "United International University · Data Analytics Laboratory · Summer 2026", 8, MINT if dark else GRAY)
        self.text(slide, 12.1, 7.11, .65, .25, str(len(self.prs.slides)), 9, MINT if dark else GRAY, align=PP_ALIGN.RIGHT)
        return slide

    def bullets(self, slide, lines, x=.65, y=1.98, w=12, size=19, color=INK, line_height=.69):
        for i, line in enumerate(lines):
            self.rect(slide, x, y + i * line_height + .1, .065, .065, TEAL)
            self.text(slide, x+.23, y + i * line_height, w-.23, line_height-.03, line, size, color)

    def image(self, slide, path, x, y, w, h):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Measured chart missing: {path}")
        with Image.open(path) as im:
            iw, ih = im.size
        scale = min(w / iw, h / ih)
        dw, dh = iw * scale, ih * scale
        slide.shapes.add_picture(str(path), Inches(x+(w-dw)/2), Inches(y+(h-dh)/2), width=Inches(dw), height=Inches(dh))

    def table(self, slide, frame, columns, labels=None, x=.65, y=2.0, w=12, h=3.2, size=13):
        frame = frame.loc[:, columns]
        table = slide.shapes.add_table(len(frame)+1, len(columns), Inches(x), Inches(y), Inches(w), Inches(h)).table
        for j, c in enumerate(columns):
            table.cell(0, j).text = (labels or {}).get(c, c.replace("_", " ").title())
        for i, (_, row) in enumerate(frame.iterrows(), start=1):
            for j, c in enumerate(columns):
                table.cell(i, j).text = _value(row[c], c)
        for i in range(len(frame)+1):
            for j in range(len(columns)):
                cell = table.cell(i, j)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor.from_string(TEAL if i == 0 else (WHITE if i % 2 else PALE))
                cell.margin_left = cell.margin_right = Inches(.07)
                cell.margin_top = cell.margin_bottom = Inches(.04)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                for p in cell.text_frame.paragraphs:
                    p.font.name = "Aptos"
                    p.font.size = Pt(size)
                    p.font.bold = i == 0
                    p.font.color.rgb = RGBColor.from_string(WHITE if i == 0 else INK)
        return table

    def note(self, slide, text):
        slide.notes_slide.notes_text_frame.text = text


def build_presentation(output_dir: Path | str = Path(".")) -> Path:
    """Read completed artifacts, produce a 14-slide PPTX and computed report."""
    root = Path(output_dir)
    quality = _load(root, "data_quality.json")
    eda = _load(root, "eda.json")
    classifier = _load(root, "classifier_metrics.json")
    feature_selection = _load(root, "feature_selection.json")
    sequence = _load(root, "sequence_metrics.json")
    budget = _load(root, "budget_metrics.json")
    clusters = _load(root, "clustering.json")
    model_comparison = _table(root, "model_comparison.csv")
    metrics = _test_rows(_table(root, "classifier_metrics.csv"))
    feature_comparison = _test_rows(_table(root, "feature_comparison.csv"))
    imbalance = _table(root, "imbalance_comparison.csv")
    importance = _table(root, "feature_importance.csv")
    seq = _sequence_test_rows(sequence)
    seq_test = next(row for row in sequence["sample_counts"] if row["split"] == "test")
    balanced = metrics.loc[metrics.operating_point.eq("balanced")].iloc[0]
    high = metrics.loc[metrics.operating_point.eq("high_recall")].iloc[0]
    optimized = budget["optimized"]
    fairness, _ = _fairness_summary(root)
    deck = _Deck()

    s = deck.slide("ReadmitRisk", "HOSPITAL READMISSION RISK & RESOURCE ALLOCATION", dark=True)
    deck.text(s, .65, 2.03, 11.8, 1.2, "From discharge risk\nto a measurable follow-up plan", 34, WHITE, True)
    deck.text(s, .68, 3.8, 11.5, .7, "Final project · Group 8 · Section DA LAB (BB)", 20, MINT)
    deck.rect(s, .65, 5.13, 12.0, 1.25, "20505B", radius=True)
    deck.text(s, .95, 5.37, 5.5, .8, "Nafiul Islam\n0152410070", 19, WHITE, True)
    deck.text(s, 7.0, 5.37, 5.3, .8, "Onika Tahmim Subha\n0152330134", 19, WHITE, True)
    deck.note(s, "All displayed numeric results are read from completed local artifacts. Approved proposal and grading criteria are in docs/.")

    s = deck.slide("A limited care budget needs a defensible priority list")
    cards = [("01", "Predict", "Estimate calibrated 30-day risk at discharge."),
             ("02", "Forecast", "Use past encounters to forecast the next encounter's outcome."),
             ("03", "Allocate", "Choose follow-up intensity under a fixed budget."),
             ("04", "Segment", "Describe patient groups and care-team workflows.")]
    for i, (number, title, body) in enumerate(cards):
        x = .65 + i * 3.15
        deck.rect(s, x, 2.02, 2.85, 3.1, WHITE, radius=True)
        deck.text(s, x+.2, 2.22, 2.4, .45, number, 22, TEAL, True)
        deck.text(s, x+.2, 2.91, 2.4, .45, title, 22, DARK, True)
        deck.text(s, x+.2, 3.6, 2.4, 1.16, body, 18)
    deck.text(s, .8, 5.57, 11.7, .96, f"Historical cohort: {quality['eligible_prevalence']:.1%} readmitted within 30 days.\nValue hypothesis: direct scarce follow-up to patients with greater expected benefit.", 21, DARK)
    deck.note(s, "Business value is simulated, not a proven clinical effect. All available discharge-time features require the encounter to be complete.")

    s = deck.slide("Data quality / Clean data and isolated patient splits")
    deck.text(s, .7, 1.96, 6.2, 1.08, f"{quality['raw_rows']:,} rows  /  {quality['raw_columns']} raw columns\n{quality['eligible_rows']:,} eligible encounters  /  {quality['eligible_patients']:,} patients", 23, DARK, True)
    deck.bullets(s, [f"Remove {quality['excluded_discharge_rows']:,} excluded discharge records; drop weight and predictor identifiers.",
                    "Keep Unknown payer/specialty; group ICD-9 codes; engineer history and medication features.",
                    "80/20 patient-grouped development/test; separate calibration and threshold validation.",
                    "Five grouped CV folds; every scaler, encoder and sampler fits training data only."], x=.75, y=3.24, w=6.0, size=16, line_height=.79)
    deck.image(s, root / "figures/01_missingness.png", 7.0, 2.02, 5.85, 3.9)
    deck.text(s, 7.12, 6.02, 5.5, .62, f"Raw predictors: {quality['raw_predictors']} after IDs/target. Meets ≥25 features and ≥2,000 rows.", 14, GRAY)
    deck.note(s, "Source: UCI Diabetes 130-US Hospitals for Years 1999–2008. Actual counts are loaded from artifacts/data_quality.json; 50 file columns and 47 predictors refer to different variable roles.")

    s = deck.slide("EDA 1 / Prior utilization identifies repeat-use patterns")
    deck.image(s, root / "figures/eda_prior_inpatient_band.png", .6, 1.9, 7.1, 4.7)
    h1 = next(h for h in eda["hypotheses"] if h["id"] == "H1")
    deck.text(s, 8.0, 2.02, 4.6, .6, f"Training target rate: {eda['readmission_rate']:.1%}", 24, TEAL, True)
    deck.bullets(s, [h1["direction_evidence"], f"H1 {h1['conclusion'].lower()}: chi-square, Holm p={h1['holm_adjusted_p']:.2g}; V={h1['effect_size']:.3f}.",
                    "Business use: flag recent repeated inpatient use for transition-of-care review."], x=8, y=2.95, w=4.75, size=17, line_height=1.12)
    deck.note(s, eda["inference_design"] + " " + eda["key_business_insights"][1])

    s = deck.slide("EDA 2 / Diagnosis and high-risk segments guide follow-up")
    deck.image(s, root / "figures/eda_diag_1_group.png", .5, 1.88, 6.25, 4.6)
    deck.image(s, root / "figures/eda_top10_risk_segments.png", 6.7, 1.88, 6.18, 4.6)
    deck.text(s, .8, 6.51, 11.9, .48, "Segments require ≥100 encounters. Their observed differences guide hypotheses, not automatic eligibility rules.", 16, DARK)
    deck.note(s, "\n".join([eda["key_business_insights"][4], eda["key_business_insights"][5], eda["key_business_insights"][9], eda["descriptive_caution"]]))

    s = deck.slide("EDA 3 / Care complexity is associated with readmission")
    deck.image(s, root / "figures/eda_length_of_stay.png", .55, 1.88, 7.3, 4.78)
    tests = pd.DataFrame(eda["hypotheses"])
    tests = tests.loc[tests.id.ne("H1"), ["id", "holm_adjusted_p", "conclusion"]].copy()
    tests["holm_adjusted_p"] = tests.holm_adjusted_p.map(lambda x: f"{x:.2g}")
    deck.table(s, tests, list(tests), {"id": "Hypothesis", "holm_adjusted_p": "Holm p"}, x=8.0, y=2.05, w=4.7, h=2.6, size=13)
    deck.text(s, 8.04, 5.0, 4.62, 1.3, "One earliest eligible encounter per patient for tests.\nInsulin/medication patterns may reflect severity; they do not prove medication harm.", 17, DARK)
    deck.note(s, "H2: one-sided Mann–Whitney U, with effect size in the report. H3: Pearson chi-square for insulin, change and diabetesMed. Five p-values are adjusted together using Holm.\n" + eda["key_business_insights"][2])

    s = deck.slide("Feature selection / Useful signal with fewer inputs")
    deck.table(s, feature_comparison, ["feature_set", "features", "accuracy", "recall", "pr_auc", "roc_auc", "training_seconds"],
               {"feature_set": "Test model", "features": "Inputs", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "training_seconds": "Fit sec"}, y=2.04, h=1.68, size=14)
    tops = ", ".join(importance.head(6)["feature"].astype(str))
    deck.text(s, .76, 4.03, 11.8, .86, f"Leading SHAP features: {tops}", 19, TEAL, True)
    deck.bullets(s, ["Cross-check SHAP with permutation importance, boosting gain and medication redundancy.",
                    f"Recommend {feature_selection['recommended'].lower()}; reduced is accepted if validation PR-AUC is within 0.005 of all features.",
                    "Local waterfall explanations describe model behavior, not causes or treatment effects."], y=5.08, size=17, line_height=.51)
    deck.note(s, "The notebook contains SHAP summary/bar/dependence plots and three local waterfall plots, plus medication correlation/VIF checks. Full feature comparison is in tables/feature_comparison.csv.")

    s = deck.slide("Four agents / Predictive results on distinct patient cohorts")
    deck.text(s, .7, 1.92, 6.05, .42, f"Agent 1 · Selected: {classifier['selected_model']}", 20, TEAL, True)
    cols = [c for c in ["model", "cv_pr_auc_mean", "cv_pr_auc_std"] if c in model_comparison]
    deck.table(s, model_comparison, cols, {"model": "Classifier", "cv_pr_auc_mean": "CV PR-AUC", "cv_pr_auc_std": "CV SD"}, x=.68, y=2.56, w=6.1, h=2.5, size=14)
    seq_balanced = seq.loc[seq.operating_point.eq("balanced")]
    deck.text(s, 7.04, 1.92, 5.58, .5, "Agent 2 · History-only test subset", 20, TEAL, True)
    deck.table(s, seq_balanced, ["model", "accuracy", "recall", "pr_auc"], {"model": "Forecaster", "pr_auc": "PR-AUC"}, x=7.0, y=2.56, w=5.7, h=2.5, size=13)
    deck.text(s, .82, 5.4, 11.75, .56, f"Agent 1 test: ROC-AUC {balanced.roc_auc:.3f} · PR-AUC {balanced.pr_auc:.3f} · Brier {balanced.brier:.3f}", 21, DARK, True)
    deck.text(s, .82, 6.07, 11.75, .72, f"Sequence majority baseline: {1-seq_test['readmission_rate']:.1%} accuracy; balanced recall remains low.\nAgents 3–4: constrained allocation + K-Means segmentation.", 16, GRAY)
    deck.note(s, "The classifier CV comparison uses training patients. Recurrent models and the historical XGBoost baseline share the same sequence target subset; their metrics must not be treated as a head-to-head comparison with all-encounter classifier metrics.")

    s = deck.slide("Imbalance / Accuracy and recall require different operating points")
    cols = [c for c in ["strategy", "cv_pr_auc_mean", "validation_recall", "validation_accuracy"] if c in imbalance]
    slide_imbalance = imbalance.copy()
    slide_imbalance["strategy"] = slide_imbalance.strategy.replace({"Natural training + threshold tuning": "Natural + tuned threshold"})
    deck.table(s, slide_imbalance, cols, {"strategy": "Training strategy", "cv_pr_auc_mean": "CV PR-AUC", "validation_recall": "Val. recall", "validation_accuracy": "Val. accuracy"},
               x=.7, y=2.0, w=7.1, h=3.45, size=13)
    deck.rect(s, 8.1, 2.0, 4.54, 1.93, PALE, radius=True)
    deck.text(s, 8.34, 2.23, 4.04, .38, "BALANCED / TEST", 14, TEAL, True)
    deck.text(s, 8.34, 2.83, 4.04, .88, f"{balanced.accuracy:.1%} accuracy\n{balanced.recall:.1%} recall · {balanced.precision:.1%} precision", 18, DARK, True)
    deck.rect(s, 8.1, 4.14, 4.54, 1.93, WHITE, radius=True)
    deck.text(s, 8.34, 4.37, 4.04, .38, "HIGH RECALL / TEST", 14, TEAL, True)
    deck.text(s, 8.34, 4.97, 4.04, .88, f"{high.recall:.1%} recall\n{high.accuracy:.1%} accuracy · {high.precision:.1%} precision", 18, DARK, True)
    deck.text(s, .85, 6.38, 11.6, .51, "Thresholds were chosen on validation; test data retain the natural class distribution. Accuracy does not measure case finding.", 16, DARK)
    deck.note(s, f"Balanced test accuracy ≥80%: {bool(balanced.accuracy >= .8)}. High-recall test recall ≥60%: {bool(high.recall >= .6)}. The validation operating rule is predeclared; observed test targets are reported honestly. SMOTE and other samplers are used only in training folds.")

    s = deck.slide("Agent 3 / Allocate follow-up under an explicit budget")
    deck.text(s, .7, 1.96, 12, .58, f"${budget['assumptions']['budget']:,.0f} budget · {optimized['patients_selected']:,} patients · {optimized['expected_prevented_readmissions']:.1f} expected readmissions prevented", 24, TEAL, True)
    budget_table = pd.DataFrame(budget["comparison"]).copy()
    budget_table["strategy"] = budget_table.strategy.replace({
        "Optimized multiple-choice": "Optimized", "Random (seed 42)": "Random", "First-come (encounter ID proxy)": "First-come",
        "Top risk (cheapest tier)": "Top risk / phone", "Blanket (all patients; may exceed budget)": "Blanket: infeasible",
        "Equal-budget lottery (expectation)": "Lottery expectation",
    })
    deck.table(s, budget_table, ["strategy", "patients_selected", "expected_prevented_readmissions", "expected_net_savings"],
               {"strategy": "Allocation", "patients_selected": "Selected", "expected_prevented_readmissions": "Expected prevented", "expected_net_savings": "Expected net savings"},
               x=.65, y=2.95, w=7.22, h=2.74, size=12)
    deck.image(s, root / "figures/budget_sensitivity.png", 8.03, 2.9, 4.77, 3.0)
    deck.text(s, .8, 6.1, 11.8, .71, f"Scenario ROI: {optimized['roi']:.1%}. Top-risk phone allocation ties the optimum under these assumptions.\nSavings are expected values, not demonstrated clinical outcomes.", 17, DARK)
    deck.note(s, _allocation_note(budget) + "\n" + json.dumps({"assumptions": budget["assumptions"], "solver_status": optimized["solution_status"], "mip_gap": optimized.get("mip_gap"), "budget_feasible": optimized["budget_feasible"]}, indent=2))

    s = deck.slide("Agent 4 / Translate clusters into care-team playbooks")
    deck.image(s, root / "figures/cluster_pca.png", .58, 1.9, 6.1, 4.67)
    profiles = pd.DataFrame(clusters["test_profiles"])
    deck.table(s, profiles, ["tier", "patients_or_encounters", "mean_prior_visits", "readmission_rate"],
               {"tier": "Tier", "patients_or_encounters": "Test n", "mean_prior_visits": "Prior visits", "readmission_rate": "Observed rate"}, x=6.98, y=2.01, w=5.7, h=2.6, size=12)
    playbooks = clusters["playbooks"]
    deck.text(s, 7.0, 4.9, 5.59, 1.54, f"Selected k={clusters['selected_k']}; silhouette={clusters['algorithm_comparison'][0]['silhouette']:.3f}.\nWeak separation: exploratory workflow groups. Care plans are descriptive suggestions, detailed in the dashboard.", 17, DARK)
    deck.note(s, json.dumps(playbooks, indent=2) + "\nCluster fitting and tier-name ranking use training information. Test profiles are descriptive checks.")

    s = deck.slide("Dashboard / One workflow for exploration and decisions")
    screenshot = next((root / p for p in ["figures/dashboard.png", "figures/dashboard_screenshot.png"] if (root / p).exists()), None)
    if screenshot:
        deck.image(s, screenshot, .65, 1.9, 8.0, 4.68)
        deck.bullets(s, ["EDA: filter age, diagnosis and admission type.", "Risk: calibrated probability, tier and local drivers.", "Budget: editable costs and effectiveness.", "Tiers: cluster profiles and playbooks."], x=8.95, y=2.12, w=3.72, size=16, line_height=1.03)
    else:
        cards = [("EDA", "Filters + KPI cards\nAge / diagnosis / admission type"), ("RISK SCORER", "Calibrated probability\nGauge + top three local SHAP drivers"),
                 ("BUDGET SIMULATOR", "Editable budget / costs / effects\nAllocation + baseline comparison"), ("RISK TIERS", "Patient map + profiles\nCare-team playbook")]
        for i, (title, description) in enumerate(cards):
            x, y = .78 + (i % 2)*6.26, 2.0 + (i // 2)*2.06
            deck.rect(s, x, y, 5.92, 1.78, WHITE, radius=True)
            deck.text(s, x+.22, y+.23, 5.5, .34, title, 17, TEAL, True)
            deck.text(s, x+.22, y+.83, 5.5, .77, description, 18)
        deck.text(s, .88, 6.47, 11.7, .38, "Launch: streamlit run app.py  ·  Sidebar: operating-point selector + model comparison", 18, DARK)
    deck.text(s, .78, 6.78, 11.9, .27, f"Dashboard: {quality['eligible_rows']:,} encounters after evaluation. EDA: {eda['encounters']:,} training encounters; {eda['patients']:,} patients for tests.", 11, GRAY)
    deck.note(s, f"The screenshot describes the post-evaluation full cleaned cohort ({quality['eligible_rows']:,} encounters). Model-development EDA is restricted to {eda['encounters']:,} training encounters, and tests use {eda['patients']:,} independent patients. Interactive exploration does not retune the frozen model. This slide describes the actual implemented Streamlit functions. If a screenshot exists at build time, it replaces the schematic description. Model resources are cached; all inputs are validated and missing fields use saved training defaults.")

    s = deck.slide("Measured results, uncertainty and next steps")
    capture = classifier.get("top20_capture")
    capture_text = f"Highest-risk 20% captures {capture:.1%} of test readmissions." if isinstance(capture, (int, float)) else f"Test PR-AUC {balanced.pr_auc:.3f} shows ranking performance under class imbalance."
    lines = [capture_text,
             f"Prioritize care complexity and repeated use; the simulated allocation yields ${optimized['expected_net_savings']:,.0f} net benefit under stated assumptions.",
             fairness[0] if fairness else "Fairness audit reports subgroup recall and selection with denominators; small groups are unstable.",
             "1999–2008 US hospitals; incomplete histories; encounter ID is a time proxy; no causal treatment-effect estimate.",
             "Next steps: external and prospective validation, local calibration and a monitored clinical workflow."]
    deck.bullets(s, lines, y=2.04, size=20, line_height=.89)
    deck.note(s, "\n".join(fairness) + "\nDo not claim a universal 0.65–0.70 ROC-AUC ceiling: achieved discrimination is a measured result, not a theoretical limit. Full fairness denominators and confidence intervals are in the saved table.")

    s = deck.slide("A reproducible analytics-to-decision prototype", "CONCLUSION / TEAM RESPONSIBILITY PLAN", dark=True)
    deck.text(s, .72, 2.06, 11.9, 1.02, "Clean data → defensible evidence → calibrated risk\n→ budget scenarios → interpretable care segments", 27, WHITE, True)
    deck.rect(s, .72, 3.57, 5.8, 1.4, "20505B", radius=True)
    deck.rect(s, 6.79, 3.57, 5.8, 1.4, "20505B", radius=True)
    deck.text(s, .98, 3.81, 5.27, .94, "Nafiul Islam · 0152410070\nClassifier + clustering", 20, WHITE, True)
    deck.text(s, 7.05, 3.81, 5.27, .94, "Onika Tahmim Subha · 0152330134\nSequence forecaster + budget allocation", 19, WHITE, True)
    deck.text(s, .9, 5.41, 11.53, .76, "Both: cleaning / EDA · explanation / model review · dashboard · testing / presentation", 20, MINT)
    deck.text(s, .9, 6.48, 11.53, .4, "Notebook + artifacts + dashboard + measured final report are ready for review.", 17, WHITE)
    deck.note(s, "Task allocation follows the approved proposal and records intended responsibilities. It does not assert independently observed individual contributions. Numeric results and limitations are reproducible from the submission; see README, criterion checklist and viva guide.")

    assert len(deck.prs.slides) == 14
    path = root / "ReadmitRisk_Final_Presentation.pptx"
    deck.prs.save(path)
    build_final_report(root)
    print(f"Phase 7 complete: {path.name} (14 slides) and docs/FINAL_REPORT.md use computed artifacts only.")
    return path


if __name__ == "__main__":
    build_presentation(Path.cwd())
