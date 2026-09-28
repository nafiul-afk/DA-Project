"""Create the submission notebook; execution is a separate, verifiable step."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# ReadmitRisk — Hospital Readmission Risk & Resource Allocation Platform
**Group 8 · Data Analytics Laboratory · Summer 2026 · United International University**  
Nafiul Islam (0152410070) · Onika Tahmim Subha (0152330134)

This notebook follows the approved proposal and all eight grading criteria. It moves from data quality and business questions to four agent roles, honest model evaluation, explanations, allocation, clustering, and a working dashboard. Model inputs are available **at discharge**, so this is not an admission-time screening model.

The supplied executed version displays real results. Run all cells from the project directory. Saved results are reused explicitly by default; set `FORCE_REBUILD = True` to train everything again. On a fresh copy without artifacts the same cells build them. The commented implementation lives in `readmitrisk/` so the notebook and app use exactly the same logic. Keep those source files with the notebook.
""")
code(r"""
import os, sys, json, random
from pathlib import Path
os.environ['PYTHONHASHSEED'] = '42'
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/readmitrisk-mpl')
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image
from run_project import run_phase, load_prepared, aggregate_metrics
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
ROOT = Path.cwd()
assert (ROOT / 'readmitrisk').is_dir(), 'Open the notebook from the project folder.'
FORCE_REBUILD = False
pd.set_option('display.max_columns', 24)
pd.set_option('display.precision', 4)

def table(name):
    frame = pd.read_csv(ROOT / 'tables' / name)
    display(frame)
    return frame

def figure(name, caption=None, width=950):
    display(Image(filename=str(ROOT / 'figures' / name), width=width))
    if caption:
        display(Markdown('**Business insight:** ' + caption))

print('Seed:', SEED, '| Python:', sys.version.split()[0])
""")
md("""
## Phase 1 — Data quality and preprocessing (criteria 1–2)

The CSV should have 101,766 rows and 50 columns. Two columns are identifiers and one is the outcome, leaving **47 raw predictors**, which exceeds the 25-feature/2,000-row requirement. The UCI ID mapping gives administrative codes their real meanings. The raw dataset and source PDFs are included under `data/` and `docs/`.

We replace `?` with missing values, drop weight, and retain payer/specialty with an `Unknown` category. Literal `None` in lab results means `Not tested`; it must not disappear during CSV parsing. Discharges in {11,13,14,19,20,21} are excluded. Repeated patients are retained as repeated encounters but remain in one partition. Duplicate encounter IDs would be removed and counted.

ICD-9 codes become the nine proposal groups; V/E codes and missing codes become Other. Fixed row-wise engineered features include age midpoint, prior visits, medication changes/active drugs, procedures per day, diagnosis diversity, and polypharmacy. Age uses its midpoint as an ordinal numeric feature, while nominal variables are one-hot encoded. Lab results are categorical, keeping the clinically distinct not-tested group.

The train/test split is **80/20 by patient**. Inside the 80% development patients we reserve separate calibration and validation groups, yielding about 60/10/10/20 overall. All learned dropping, rare-category pooling, imputation, encoding, scaling and resampling happen inside a pipeline fitted only on the appropriate training patients. IDs and targets never enter the feature matrix.
""")
code(r"""
quality = run_phase('data', force=FORCE_REBUILD)
df, splits = load_prepared(ROOT)
from readmitrisk.data import feature_columns, diagnosis_group, MEDICATIONS
display(pd.DataFrame([{k: v for k, v in quality.items() if isinstance(v, (int, float, str))}]))
split_summary = table('split_summary.csv')
assert quality['raw_predictors'] >= 25 and quality['raw_rows'] >= 2000
assert not {'patient_nbr', 'encounter_id', 'readmit_30', 'readmitted'} & set(feature_columns(df))
for left, a in splits.items():
    for right, b in splits.items():
        if left != right:
            assert set(df.iloc[a].patient_nbr).isdisjoint(df.iloc[b].patient_nbr)
print('Patient overlap: 0. Candidate features:', quality['candidate_predictors'])
print('Train-fitted near-zero columns:', ', '.join(quality['train_near_zero_columns']))
table('missingness.csv')
figure('01_missingness.png')
""")
code(r"""
# Small transparent examples help explain the deterministic ICD-9 logic in a viva.
examples = ['250.83', '401', '486', '786.5', 'V45', 'E885', '?']
display(pd.DataFrame({'ICD9': examples, 'group': [diagnosis_group(c) for c in examples]}))
display(df[['age', 'age_ordinal', 'total_prior_visits', 'med_changes_count',
            'meds_active_count', 'lab_procs_per_day', 'diag_diversity', 'polypharmacy']].head())
""")
md("""
## Phase 2 — EDA and business hypotheses (criterion 3)

EDA uses **training patients only**, preserving the independent test. Charts describe encounters; confidence in an association must account for repeat patients. Therefore the formal tests use each patient's earliest eligible encounter, ordered by encounter ID. That makes the test rows independent across patients, subject to unmeasured hospital clustering.

**H1:** more prior inpatient visits are associated with higher readmission (chi-square, plus direction check). **H2:** readmitted patients tend to have longer stays (one-sided Mann–Whitney U). **H3:** insulin status, medication change, and diabetes-medication use are associated with readmission (three chi-square tests). Holm adjustment controls the family-wise error rate across all five tests. A small p-value is not an effect size or proof of causation.
""")
code(r"""
eda = run_phase('eda', force=FORCE_REBUILD)
print(eda['cohort'])
for chart in eda['figures']:
    display(Markdown('### ' + chart['title']))
    figure(Path(chart['figure']).name, chart['insight'])
""")
code(r"""
display(pd.DataFrame(eda['hypotheses'])[['id', 'test', 'n_patients', 'p_value',
    'holm_adjusted_p', 'effect_size_name', 'effect_size', 'conclusion']])
display(Markdown(eda['inference_design']))
display(Markdown('### Key Business Insights\n\n' + '\n'.join('- ' + x for x in eda['key_business_insights'])))
display(pd.DataFrame(eda['top_risk_segments']))
""")
md("""
## Phase 3 — Agent 1: Readmission Risk Classifier (criterion 6)

We compare Logistic Regression, Random Forest, XGBoost and LightGBM using five-fold `StratifiedGroupKFold`. Randomized search tunes both boosting families with **average precision (AP, reported as PR-AUC)**, not accuracy. AP summarizes ranking quality relative to the positive-class prevalence.

Class weighting, SMOTE, BorderlineSMOTE and random undersampling are compared on the same training folds with the same LightGBM settings. Synthetic sampling follows fold-fitted encoding/scaling; its fractional dummy variables are an acknowledged limitation. Test/validation records are never resampled. The final estimator is selected from the natural/class-weighted tuned family; the resampling table is a controlled comparison.

`CalibratedClassifierCV(FrozenEstimator(...), method='sigmoid')` fits calibration using separate patients. Validation selects two thresholds: maximum recall subject to **82% validation accuracy** (a predeclared margin for the 80% test requirement), and maximum precision subject to **70% validation recall**. Neither requirement is guaranteed on new patients. All choices are locked before scoring the test.

Reported calibrated CV uses fold-local fitting/calibration/thresholds. Hyperparameter and reduced-feature selection are not nested inside an outer search, so CV is explicitly **post-selection** and may be optimistic; the independent test is the primary assessment. Training metrics are resubstitution estimates and also optimistic.
""")
code(r"""
classifier = run_phase('classifier', force=FORCE_REBUILD)
model_comparison = table('model_comparison.csv')
imbalance_comparison = table('imbalance_comparison.csv')
display(Markdown('**Final estimator:** ' + classifier['selected_model'] +
                 ' · **Feature set:** ' + classifier['recommended_features']))
classifier_metrics = table('classifier_metrics.csv')
print('Majority-only held-out accuracy:', f"{classifier['majority_baseline_accuracy']:.2%}")
print('Held-out 80% accuracy requirement met:', classifier['accuracy_requirement_met'])
print('Held-out high-recall requirement (at least 60%) met:', classifier['high_recall_requirement_met'])
display(Markdown(classifier['cv_note']))
""")
code(r"""
figure('classifier_confusion.png')
figure('classifier_roc_pr_calibration.png')
figure('classifier_lift_gain.png', f"The highest-risk 20% captures {classifier['top20_capture']:.1%} of held-out readmissions.")
table('classifier_deciles.csv')
table('classifier_cv_folds.csv')
""")
md("""
## Agent 2 — Sequential Risk Forecaster (criterion 6)

For patients with at least two eligible encounters, the last up to three **prior encounters** predict the next observed encounter's `readmit_30`. Shorter histories are padded and masked. The target encounter's features and all previous outcome labels are excluded. `encounter_id` only orders the records; no real timestamps or elapsed times are available.

GRU and LSTM use class weights, dropout, early stopping and learning-rate reduction, with the same patient partitions as Agent 1. A history-only XGBoost baseline receives the last prior encounter, history mean, and history length and is evaluated on **exactly the same targets**. This is a different prediction problem from contemporaneous Agent 1 scoring; their headline metrics should not be compared as if the input information were identical.

Five-fold grouped CV refits preprocessing and internal calibration/threshold selection inside each fold. Recurrent CV has an eight-epoch cap; final training has a fifteen-epoch cap. Actual trained epochs are recorded. This forecasting cohort is selected: it contains patients who have a later observed encounter. It does not predict whether or when an encounter will occur.
""")
code(r"""
sequences = run_phase('sequences', force=FORCE_REBUILD)
display(pd.DataFrame(sequences['sample_counts']))
sequence_rows = []
for model, result in sequences['models'].items():
    for split, points in result['metrics'].items():
        for point, metrics in points.items():
            sequence_rows.append(dict(model=model, split=split, operating_point=point, **metrics))
sequence_results = pd.DataFrame(sequence_rows)
display(sequence_results)
display(pd.DataFrame(sequences['cross_validation']['scores']).groupby(['model', 'operating_point'])[
    ['accuracy','balanced_accuracy','precision','recall','f1','roc_auc','pr_auc','brier']].mean())
display(Markdown('**Selected using validation PR-AUC:** ' + sequences['selected_by_validation_pr_auc']))
for note in sequences['limitations']:
    display(Markdown('- ' + note))
""")
code(r"""
figure('sequence_training_curves.png')
figure('sequence_confusion_matrices.png')
figure('sequence_diagnostics.png')
# Same deciles and capture definition for all three history models.
table('sequence_lift_gain.csv')
""")
md("""
## Agent 3 — Follow-Up Budget Allocator (criterion 6: value creation)

Each held-out patient enters once, using their latest eligible encounter and its calibrated probability. We solve a multiple-choice binary knapsack: choose at most one intervention per patient and keep total spending within budget. Expected prevented readmissions are `risk × assumed relative risk reduction`; net savings subtract intervention spending from avoided readmission costs; ROI is net savings divided by spending.

Defaults: phone call $50 /15% effectiveness; home visit $250 /25%; case manager $600 /35%; readmission $15,000. The default budget funds phone calls for about 10% of eligible test patients. These are **editable scenario assumptions, not causal effects or demonstrated savings**. The solver reports its status and remaining optimality gap. A time-limited result is not falsely called an exact optimum.

Compare random, first-come, top-risk, and blanket allocation. Blanket follow-up for every patient may exceed budget and is labeled infeasible. An equal-budget lottery expectation supplies a feasible blanket-style comparison. With homogeneous costs/effects, top-risk phone calls may match the optimizer; that is an interpretable result, not a bug.
""")
code(r"""
allocation = run_phase('allocation', force=FORCE_REBUILD)
display(allocation['assumptions'])
table('budget_comparison.csv')
table('budget_sensitivity.csv')
for path in sorted((ROOT / 'figures').glob('budget*.png')):
    figure(path.name)
print('Solver result:', allocation['optimized'])
""")
md("""
## Agent 4 — Patient Risk-Tier Clustering (criterion 6: value creation)

Train-only imputation/scaling prepares selected numeric features. We compare k=2…6 using elbow distance, silhouette and Davies–Bouldin, then compare the chosen K-Means model with GaussianMixture. Outcomes do not enter clustering; training readmission rates are used only afterwards to order tier names. PCA is fitted on training data and visualizes held-out encounters.

Low silhouette scores mean substantial overlap. These tiers describe patterns and suggest a care-team review workflow; they are not validated clinical categories. Held-out profiles report actual rates, age, stay, prior visits, medications and dominant diagnosis.
""")
code(r"""
clustering = run_phase('clustering', force=FORCE_REBUILD)
table('cluster_selection.csv')
table('cluster_algorithm_comparison.csv')
table('cluster_profiles.csv')
table('cluster_playbooks.csv')
figure('cluster_selection.png')
figure('cluster_pca.png')
""")
md("""
## Phase 4 — Feature selection and explanations (criterion 4)

Tree SHAP on training patients ranks original features by summing absolute importance over their one-hot columns. The top18 nonredundant original features define the reduced model. Validation permutation importance and XGBoost gain provide complementary views; they do not replace the train-only selection rule. Correlation/VIF checks cover all 23 medication active flags, reporting constants separately and removing highly correlated lower-ranked flags from the reduced candidate.

The all-feature and reduced models are compared on accuracy, recall, F1, PR-AUC, ROC-AUC, training time and retained feature count. We choose the reduced variant only if its validation AP is within 0.005 of the full model. Test numbers report that locked decision rather than choose it.

SHAP plots explain the **base estimator's raw output**, usually log odds. They do not decompose the final sigmoid-calibrated probability. The three local waterfalls are training examples for explanation, not evidence of clinical correctness.
""")
code(r"""
feature_selection = json.loads((ROOT / 'artifacts/feature_selection.json').read_text())
display(Markdown('**Recommendation:** ' + feature_selection['recommended'] + '. ' + feature_selection['selection_rule']))
table('feature_comparison.csv')
table('feature_importance.csv')
table('permutation_importance.csv')
table('xgboost_gain.csv')
table('medication_vif.csv')
table('redundant_medication_pairs.csv')
figure('medication_correlation.png')
figure('shap_summary.png')
figure('shap_bar.png')
figure('shap_dependence.png')
for i in range(1,4):
    figure(f'shap_waterfall_patient_{i}.png', width=900)
""")
md("""
## Phase 5 — Fairness and Summary of Key Insights (criterion 5)

Recall and selection rate are audited by race, gender and age at both operating points. Denominators, positive counts, Wilson recall intervals and small-group flags make uncertainty visible. Repeat encounters and unobserved hospital effects mean these descriptive intervals are not a complete inferential fairness analysis. Disparities may reflect sampling, access, measurement and disease burden; race is not a biological causal explanation.

The data covers US hospitals in 1999–2008. Current practice, populations, coding, social circumstances and readmission capture differ. A quoted 0.65–0.70 AUROC benchmark is context, not a mathematical ceiling; our measured values below take precedence. Prospective local validation and intervention evaluation are required before operational use.
""")
code(r"""
fairness = table('fairness.csv')
figure('fairness.png')
test_balanced = classifier_metrics.query("split == 'test' and operating_point == 'balanced'").iloc[0]
test_high = classifier_metrics.query("split == 'test' and operating_point == 'high_recall'").iloc[0]
top_features = pd.read_csv(ROOT / 'tables/feature_importance.csv').head(5).feature.tolist()
summary = [
    f"The strongest model signals include {', '.join(top_features)}; these are predictive associations.",
    f"At the locked balanced point, accuracy is {test_balanced.accuracy:.2%}, recall {test_balanced.recall:.2%} and precision {test_balanced.precision:.2%}.",
    f"At the high-recall point, recall is {test_high.recall:.2%} with accuracy {test_high.accuracy:.2%}; capacity determines the usable operating point.",
    f"The top 20% of risk scores captures {classifier['top20_capture']:.1%} of held-out readmissions.",
    f"The selected feature set is {feature_selection['recommended']}; the choice was made with validation results.",
    "Prior inpatient use and transitions of care suggest who to prioritize for review, rather than automatic treatment changes.",
    "History forecasting meets the accuracy target with low recall; its high-recall alternatives have large false-positive workloads.",
    "Budget estimates are scenario outputs under fixed intervention-effect assumptions, not observed reductions in readmissions.",
    "Risk clusters overlap, and fairness results need small-group uncertainty and local validation before care decisions.",
]
display(Markdown('### Summary of Key Insights\n\n' + '\n'.join('- ' + x for x in summary)))
display(pd.DataFrame([allocation['optimized']]))
""")
md("""
## Phase 6 — Interactive Streamlit dashboard (criterion 7)

Run `streamlit run app.py` in the activated environment. Four tabs match the proposal:

1. **EDA:** age, diagnosis and admission filters; KPIs and Plotly charts.
2. **Risk Scorer:** editable discharge profile, calibrated risk gauge, cluster tier and three SHAP drivers; missing fields use training defaults.
3. **Budget Simulator:** editable budget, tier costs/effects and readmission cost; constrained solver and baseline comparisons.
4. **Risk Tiers:** PCA scatter, held-out profiles and care-team playbook.

The sidebar shows the model comparison and switches the accuracy/recall operating point. Resources and data are cached; optimization runs on form submission. Changing a threshold changes the action flag, not the patient's calibrated risk probability.
""")
code(r"""
for name in ['dashboard.png', 'dashboard_risk.png', 'dashboard_budget.png', 'dashboard_tiers.png']:
    screenshot = ROOT / 'figures' / name
    if screenshot.exists():
        figure(screenshot.name)
print('Dashboard command: streamlit run app.py')
""")
md("""
## Phase 7 — Presentation, grading map and viva (criterion 8)

The presentation uses the saved measurements directly. It has 14 slides in the proposal's teal/dark theme. The responsibility split follows the proposal and should be confirmed by the team; it is a plan, not a claim about work performed by either student.
""")
code(r"""
all_metrics = aggregate_metrics(ROOT)
run_phase('report', root=ROOT)
display(Markdown((ROOT / 'docs/CRITERIA_CHECKLIST.md').read_text()))
display(Markdown((ROOT / 'docs/VIVA_QUESTIONS.md').read_text()))
print('Presentation:', ROOT / 'ReadmitRisk_Final_Presentation.pptx')
print('Metrics:', ROOT / 'artifacts/metrics.json')
print('Figures:', len(list((ROOT / 'figures').glob('*.png'))))
print('CSV tables:', len(list((ROOT / 'tables').glob('*.csv'))))
""")
md("""
## Sources and implementation references

- [UCI dataset and CC BY4.0 attribution](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008): Clore, Cios, DeShazo and Strack (2014), DOI10.24432/C5230J.
- [Scikit-learn probability calibration](https://scikit-learn.org/1.6/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).
- [Imbalanced-learn leakage pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html).
- [SciPy mixed integer optimization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).
- Approved project proposal and official grading criteria: `docs/ReadmitRisk_Project_Proposal.pdf`, `docs/Project_Criteria_docx.pdf`.

All numerical statements in this submission come from the CSV and the executed code. Cost and intervention-effect values are explicitly scenario assumptions.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.update(kernelspec={"display_name": "Python 3 (ReadmitRisk)", "language": "python", "name": "python3"},
                         language_info={"name": "python", "version": "3.12.3"})
nbf.write(notebook, ROOT / "ReadmitRisk_Notebook.ipynb")
print(f"Created notebook with {len(cells)} cells.")
