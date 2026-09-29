"""ReadmitRisk dashboard. Run with: streamlit run app.py."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from readmitrisk.allocation import compare_allocations, patient_allocation_cohort
from readmitrisk.data import MEDICATIONS

ROOT = Path(__file__).resolve().parent
TEAL = "#17b8aa"
COLORS = ["#17b8aa", "#58c4d9", "#edbd61", "#eb835f", "#cf688c", "#7d80ca"]
px.defaults.color_discrete_sequence = COLORS
DIAGNOSES = ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"]

st.set_page_config(page_title="ReadmitRisk · Group 8", page_icon="🏥", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 4rem; padding-bottom: 3rem; max-width: 1500px;}
h1, h2, h3 {letter-spacing: -.035em;}
[data-testid="stMetric"] {background: #132a32; padding: 1rem; border-radius: 10px; border: 1px solid #24454d;}
[data-testid="stMetricLabel"] {color: #a8cbd0;}
.eyebrow {color: #17b8aa; letter-spacing: .15em; font-size: .8rem; font-weight: 700;}
.subtitle {color: #a8cbd0; max-width: 980px; margin-bottom: 1.6rem;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_json(name, modified=0):
    path = ROOT / name
    return json.loads(path.read_text()) if path.exists() else {}


def json_artifact(name):
    path = ROOT / name
    return load_json(name, path.stat().st_mtime_ns if path.exists() else 0)


@st.cache_data(show_spinner=False)
def load_csv(name, modified=0):
    return pd.read_csv(ROOT / name, low_memory=False)


def csv_artifact(name):
    path = ROOT / name
    return load_csv(name, path.stat().st_mtime_ns) if path.exists() else pd.DataFrame()


@st.cache_resource(show_spinner="Loading the saved model…")
def load_model(name, modified=0):
    return joblib.load(ROOT / name)


def model_artifact(name):
    path = ROOT / name
    return load_model(name, path.stat().st_mtime_ns)


def styled_chart(fig, height=350):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=10, r=20, t=50, b=15),
                      colorway=COLORS, font=dict(family="Arial", color="#dcecef"))
    return fig


def percent_text(value, digits=1):
    """Format a probability consistently, including values read as object dtype."""
    return "—" if pd.isna(value) else f"{float(value):.{digits}%}"


def decimal_text(value, digits=3):
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def metrics_display(frame, percent_columns=(), decimal_columns=(), seconds_columns=()):
    """Use strings for dashboard tables so Streamlit never exposes raw fractions."""
    view = frame.copy()
    for column in percent_columns:
        if column in view:
            view[column] = view[column].map(percent_text)
    for column in decimal_columns:
        if column in view:
            view[column] = view[column].map(decimal_text)
    for column in seconds_columns:
        if column in view:
            view[column] = view[column].map(lambda value: "—" if pd.isna(value) else f"{float(value):.1f}")
    return view


def display_comparison(table):
    visible = [col for col in ["strategy", "patients_selected", "spending", "budget_feasible",
                               "expected_prevented_readmissions", "expected_net_savings", "roi"] if col in table]
    st.dataframe(table[visible], hide_index=True, use_container_width=True,
                 column_config={"spending": st.column_config.NumberColumn("Spending ($)", format="$%.0f"),
                                "expected_prevented_readmissions": st.column_config.NumberColumn("Expected prevented", format="%.2f"),
                                "expected_net_savings": st.column_config.NumberColumn("Expected net savings ($)", format="$%.0f"),
                                "roi": st.column_config.NumberColumn("Net ROI (multiple)", format="%.2f×")})


def numeric_default(defaults, name, fallback, minimum=0, maximum=100):
    value = defaults.get(name, fallback)
    try:
        return int(np.clip(float(value), minimum, maximum))
    except (TypeError, ValueError):
        return int(fallback)


def categorical_input(label, choices, default, key):
    choices = list(dict.fromkeys(list(choices) + ([str(default)] if default is not None else [])))
    index = choices.index(str(default)) if str(default) in choices else 0
    return st.selectbox(label, choices, index=index, key=key)


def model_threshold(metadata, operating_point):
    thresholds = metadata.get("thresholds", {})
    value = thresholds.get(operating_point, .5)
    if isinstance(value, dict):
        value = value.get("threshold", .5)
    return float(value)


def local_shap(row):
    """Explain the uncalibrated tree score; calibration does not preserve SHAP sums."""
    import shap
    pipeline = model_artifact("artifacts/base_pipeline.joblib")
    transformed = row
    # Resamplers are fit-time operations; never apply them to a dashboard record.
    for name, step in pipeline.steps[:-1]:
        if hasattr(step, "transform"):
            transformed = step.transform(transformed)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    estimator = pipeline.steps[-1][1]
    explainer = (model_artifact("artifacts/shap_explainer.joblib")
                 if (ROOT / "artifacts/shap_explainer.joblib").exists()
                 else shap.TreeExplainer(estimator))
    explanation = explainer(transformed)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    contributions = values[0]
    top = np.argsort(-np.abs(contributions))[:3]
    labels = {
        "number_inpatient": "Prior inpatient visits", "number_emergency": "Prior emergency visits",
        "number_outpatient": "Prior outpatient visits", "total_prior_visits": "Total prior visits",
        "age_ordinal": "Age band midpoint", "time_in_hospital": "Hospital stay",
        "num_medications": "Number of medications", "num_lab_procedures": "Lab procedures",
        "lab_procs_per_day": "Lab procedures per day", "med_changes_count": "Drugs with dose changes",
        "meds_active_count": "Active diabetes drugs", "number_diagnoses": "Number of diagnoses",
        "diag_1_group": "Primary diagnosis", "diag_2_group": "Secondary diagnosis", "diag_3_group": "Third diagnosis",
    }
    def readable_feature(value):
        value = str(value).split("__", 1)[-1]
        for name in sorted(row.columns, key=len, reverse=True):
            title = labels.get(name, name.replace("_", " ").capitalize())
            if value == name:
                return title
            if value.startswith(name + "_"):
                return f"{title}: {value[len(name)+1:]}"
        return value.replace("_", " ")
    table = pd.DataFrame({"Feature": [readable_feature(name) for name in np.asarray(names)[top]], "SHAP contribution": contributions[top]})
    model_name = type(estimator).__name__
    scale = "uncalibrated probability" if model_name in {"RandomForestClassifier", "ExtraTreesClassifier", "DecisionTreeClassifier"} else "uncalibrated log-odds"
    return table, scale


@st.cache_data(show_spinner=False, max_entries=12)
def simulate_budget(cohort, budget, costs, effects, readmission_cost, objective):
    return compare_allocations(cohort, budget, costs, effects, readmission_cost,
                               objective=objective, time_limit=12)


metadata = json_artifact("artifacts/model_metadata.json")
eda = json_artifact("artifacts/eda.json")
quality = json_artifact("artifacts/data_quality.json")
data = csv_artifact("data/cleaned_data.csv")

with st.sidebar:
    st.markdown("### ReadmitRisk")
    st.caption("GROUP 8 · DATA ANALYTICS LAB")
    st.markdown("**United International University**  \nSummer 2026")
    st.divider()
    operating_label = st.radio("Classifier operating point", ["Balanced / accuracy", "High recall"],
                               help="Thresholds were selected using validation patients. Changing the threshold changes alerts, not calibrated probabilities.")
    operating = "balanced" if operating_label.startswith("Balanced") else "high_recall"
    threshold = model_threshold(metadata, operating)
    if metadata:
        st.caption(f"Alert threshold: {threshold:.1%} calibrated risk")
        st.markdown(f"Selected model: **{metadata.get('selected_model', metadata.get('model_name', 'See comparison'))}**")
    comparison = csv_artifact("tables/model_comparison.csv")
    if not comparison.empty:
        st.markdown("**Model comparison**")
        keep = [col for col in comparison if col.lower() in {"model", "model_name", "cv_pr_auc_mean", "pr_auc", "cv_pr_auc", "mean_cv_pr_auc"}]
        st.dataframe(comparison[keep] if keep else comparison, hide_index=True, use_container_width=True,
                     column_config={"model": "Model", "cv_pr_auc_mean": st.column_config.NumberColumn("CV PR-AUC", format="%.3f"),
                                    "cv_roc_auc_mean": st.column_config.NumberColumn("CV ROC-AUC", format="%.3f")})
        st.caption("Five-fold patient-grouped cross-validation on training patients.")
    st.divider()
    st.caption("Nafiul Islam · 0152410070  \nOnika Tahmim Subha · 0152330134")
    st.caption("Historical US hospital data, 1999–2008. This dashboard is a university research demonstration.")

st.markdown('<div class="eyebrow">HOSPITAL ANALYTICS & CARE PLANNING</div>', unsafe_allow_html=True)
st.title("ReadmitRisk")
st.markdown('<div class="subtitle">Understand 30-day readmission risk, explore patient groups, and estimate the value of follow-up care within a fixed budget.</div>', unsafe_allow_html=True)
if data.empty:
    st.info("Generate the analysis and saved artifacts first: run `python run_project.py` from the project folder. Place `diabetic_data.csv` and `IDS_mapping.csv` in `data/` before running.")
    st.stop()

tab_eda, tab_risk, tab_budget, tab_tiers, tab_metrics = st.tabs(
    ["01 · EDA", "02 · Risk Scorer", "03 · Budget Simulator", "04 · Risk Tiers", "05 · Model Metrics"]
)

with tab_eda:
    st.subheader("Explore the readmission picture")
    st.caption("Eligible encounters exclude expired and hospice discharge codes. Filters update the descriptive charts; these associations do not establish causation.")
    columns = st.columns(3)
    filter_fields = [("age", "Age band"), ("diag_1_group", "Primary diagnosis"), ("admission_type", "Admission type")]
    filtered = data.copy()
    for column, (field, label) in zip(columns, filter_fields):
        if field in data:
            options = sorted(data[field].dropna().astype(str).unique())
            with column:
                selected = st.multiselect(label, options, default=[], placeholder="All", key=f"filter_{field}")
            if selected:
                filtered = filtered[filtered[field].astype(str).isin(selected)]
    kpis = st.columns(4)
    kpis[0].metric("Eligible encounters", f"{len(filtered):,}")
    kpis[1].metric("Unique patients", f"{filtered.patient_nbr.nunique():,}")
    kpis[2].metric("30-day readmission", f"{filtered.readmit_30.mean():.1%}" if len(filtered) else "—")
    kpis[3].metric("Average hospital stay", f"{filtered.time_in_hospital.mean():.1f} days" if len(filtered) else "—")
    if len(filtered):
        left, right = st.columns(2)
        age_rates = filtered.groupby("age", observed=True).readmit_30.agg(["mean", "count"]).reset_index()
        fig = px.bar(age_rates, x="age", y="mean", hover_data=["count"], title="Readmission rate by age", color_discrete_sequence=[TEAL])
        fig.update_yaxes(tickformat=".0%", title="30-day readmission rate")
        fig.update_xaxes(title="Age band")
        left.plotly_chart(styled_chart(fig), use_container_width=True)
        diagnosis_rates = filtered.groupby("diag_1_group", observed=True).readmit_30.agg(["mean", "count"]).reset_index().sort_values("mean")
        fig = px.bar(diagnosis_rates, x="mean", y="diag_1_group", orientation="h", hover_data=["count"], title="Readmission by primary diagnosis", color_discrete_sequence=[TEAL])
        fig.update_xaxes(tickformat=".0%", title="30-day readmission rate")
        fig.update_yaxes(title=None)
        right.plotly_chart(styled_chart(fig), use_container_width=True)
        los = filtered.assign(Outcome=np.where(filtered.readmit_30 == 1, "Readmitted <30 days", "Not readmitted <30 days"))
        left, right = st.columns(2)
        fig = px.histogram(los, x="time_in_hospital", color="Outcome", barmode="overlay", histnorm="percent", nbins=14,
                           opacity=.7, title="Length of stay by outcome")
        fig.update_xaxes(title="Hospital days")
        left.plotly_chart(styled_chart(fig), use_container_width=True)
        prior = filtered.assign(prior_group=filtered.number_inpatient.clip(upper=4).astype(str).replace("4", "4+"))
        prior = prior.groupby("prior_group", observed=True).readmit_30.agg(["mean", "count"]).reset_index()
        fig = px.bar(prior, x="prior_group", y="mean", hover_data=["count"], title="Prior inpatient visits and readmission", color_discrete_sequence=[TEAL])
        fig.update_yaxes(tickformat=".0%", title="Readmission rate")
        fig.update_xaxes(title="Prior inpatient visits")
        right.plotly_chart(styled_chart(fig), use_container_width=True)
    else:
        st.info("No encounters match this filter combination. Clear a filter to broaden the cohort.")
    if eda.get("key_business_insights"):
        with st.expander("Computed business insights from the training cohort"):
            for insight in eda["key_business_insights"]:
                st.markdown(f"- {insight}")
    if eda.get("hypotheses"):
        with st.expander("Hypothesis tests and multiple-testing correction"):
            st.dataframe(pd.DataFrame(eda["hypotheses"]), hide_index=True, use_container_width=True)

with tab_risk:
    st.subheader("Score a discharge profile")
    st.caption("The saved classifier estimates readmission within 30 days. Unspecified fields use training-set defaults; adjust the profile before scoring.")
    ready = all((ROOT / name).exists() for name in ["artifacts/classifier.joblib", "artifacts/feature_defaults.json", "artifacts/model_metadata.json"])
    if not ready:
        st.info("The classifier is not saved yet. Finish `python run_project.py`, then refresh this page.")
    else:
        defaults = json_artifact("artifacts/feature_defaults.json")
        if "defaults" in defaults and isinstance(defaults["defaults"], dict):
            defaults = defaults["defaults"]
        with st.form("risk_profile"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age_default = f"[{numeric_default(defaults, 'age_ordinal', 65, 5, 95) // 10 * 10}-{numeric_default(defaults, 'age_ordinal', 65, 5, 95) // 10 * 10 + 10})"
                age_band = categorical_input("Age band", [f"[{i}-{i+10})" for i in range(0, 100, 10)], age_default, "score_age")
                los = st.number_input("Hospital stay (days)", 1, 14, numeric_default(defaults, "time_in_hospital", 4, 1, 14))
                meds = st.number_input("Number of medications", 0, 100, numeric_default(defaults, "num_medications", 16, 0, 100))
                labs = st.number_input("Laboratory procedures", 0, 150, numeric_default(defaults, "num_lab_procedures", 44, 0, 150))
            with c2:
                inpatient = st.number_input("Prior inpatient visits", 0, 30, numeric_default(defaults, "number_inpatient", 0, 0, 30))
                emergency = st.number_input("Prior emergency visits", 0, 80, numeric_default(defaults, "number_emergency", 0, 0, 80))
                outpatient = st.number_input("Prior outpatient visits", 0, 50, numeric_default(defaults, "number_outpatient", 0, 0, 50))
                diagnosis = categorical_input("Primary diagnosis", DIAGNOSES, defaults.get("diag_1_group", "Circulatory"), "score_diagnosis")
            with c3:
                a1c = categorical_input("A1C result", ["Not tested", "Norm", ">7", ">8"], defaults.get("A1Cresult", "Not tested"), "score_a1c")
                glucose = categorical_input("Maximum glucose result", ["Not tested", "Norm", ">200", ">300"], defaults.get("max_glu_serum", "Not tested"), "score_glucose")
                insulin = categorical_input("Insulin status", ["No", "Steady", "Up", "Down"], defaults.get("insulin", "No"), "score_insulin")
                metformin = categorical_input("Metformin status", ["No", "Steady", "Up", "Down"], defaults.get("metformin", "No"), "score_metformin")
            with st.expander("Admission context and additional information"):
                a, b, c = st.columns(3)
                with a:
                    admission = categorical_input("Admission type", sorted(set(data.admission_type.dropna()) | {"Unknown"}), defaults.get("admission_type", "Emergency"), "score_admission")
                    race = categorical_input("Race (historical dataset category)", sorted(data.race.dropna().unique()), defaults.get("race", "Unknown"), "score_race")
                    diag2 = categorical_input("Secondary diagnosis", DIAGNOSES, defaults.get("diag_2_group", "Other"), "score_diag2")
                with b:
                    discharge = categorical_input("Discharge disposition", sorted(set(data.discharge_disposition.dropna()) | {"Unknown"}), defaults.get("discharge_disposition", "Unknown"), "score_discharge")
                    gender = categorical_input("Gender (historical dataset category)", sorted(data.gender.dropna().unique()), defaults.get("gender", "Unknown"), "score_gender")
                    diag3 = categorical_input("Third diagnosis", DIAGNOSES, defaults.get("diag_3_group", "Other"), "score_diag3")
                with c:
                    source = categorical_input("Admission source", sorted(set(data.admission_source.dropna()) | {"Unknown"}), defaults.get("admission_source", "Unknown"), "score_source")
                    procedures = st.number_input("Number of procedures", 0, 20, numeric_default(defaults, "num_procedures", 0, 0, 20))
                    diagnoses = st.number_input("Number of diagnoses", 1, 20, numeric_default(defaults, "number_diagnoses", 7, 1, 20))
            submitted = st.form_submit_button("Calculate readmission risk", type="primary", use_container_width=True)
        if submitted:
            row = dict(defaults)
            row.update(age_ordinal=int(age_band.split("-")[0].strip("[")) + 5, time_in_hospital=los,
                       num_medications=meds, num_lab_procedures=labs, number_inpatient=inpatient,
                       number_emergency=emergency, number_outpatient=outpatient, diag_1_group=diagnosis,
                       diag_2_group=diag2, diag_3_group=diag3, A1Cresult=a1c, max_glu_serum=glucose,
                       insulin=insulin, metformin=metformin, admission_type=admission,
                       discharge_disposition=discharge, admission_source=source, race=race, gender=gender,
                       num_procedures=procedures, number_diagnoses=diagnoses, total_prior_visits=inpatient + emergency + outpatient,
                       lab_procs_per_day=labs / los, polypharmacy=int(meds > 15), diag_diversity=len({diagnosis, diag2, diag3}))
            medication_statuses = [row.get(name, "No") for name in MEDICATIONS]
            row["med_changes_count"] = sum(value in {"Up", "Down"} for value in medication_statuses)
            row["meds_active_count"] = sum(value != "No" for value in medication_statuses)
            row["change"] = "Ch" if row["med_changes_count"] else "No"
            row["diabetesMed"] = "Yes" if row["meds_active_count"] else "No"
            columns = metadata.get("feature_columns", list(defaults))
            full_frame = pd.DataFrame([row])
            frame = full_frame.reindex(columns=columns)
            with st.spinner("Computing calibrated risk and the individual explanation…"):
                probability = float(model_artifact("artifacts/classifier.joblib").predict_proba(frame)[0, 1])
                st.session_state["risk_prediction"] = {"probability": probability, "row": frame, "cluster_row": full_frame}
        if "risk_prediction" in st.session_state:
            probability = st.session_state.risk_prediction["probability"]
            frame = st.session_state.risk_prediction["row"]
            left, right = st.columns([1, 1.25])
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=100 * probability,
                number={"suffix": "%", "valueformat": ".1f"}, title={"text": "Calibrated 30-day risk"},
                gauge={"axis": {"range": [0, 100], "ticksuffix": "%"}, "bar": {"color": TEAL},
                       "steps": [{"range": [0, threshold * 100], "color": "#1b504c"}, {"range": [threshold * 100, 100], "color": "#654334"}],
                       "threshold": {"line": {"color": "#ffd28b", "width": 4}, "thickness": .8, "value": threshold * 100}}))
            styled_chart(gauge, 310)
            gauge.update_layout(margin=dict(l=55, r=55, t=50, b=20))
            left.plotly_chart(gauge, use_container_width=True)
            alert = probability >= threshold
            right.metric("Decision support alert", "Review recommended" if alert else "Below selected threshold")
            right.caption(f"{operating_label} operating point · alert threshold {threshold:.1%}. The threshold selects patients for review; it does not change risk estimates.")
            if (ROOT / "artifacts/clustering.joblib").exists():
                cluster_art = model_artifact("artifacts/clustering.joblib")
                cluster_features = st.session_state.risk_prediction["cluster_row"].reindex(columns=cluster_art["features"])
                cluster_num = int(cluster_art["model"].predict(cluster_art["preprocessor"].transform(cluster_features))[0])
                right.metric("Patient profile tier", cluster_art["cluster_to_tier"][cluster_num])
                right.caption("Tier is the nearest unsupervised patient profile, named using training readmission rates.")
            if (ROOT / "artifacts/base_pipeline.joblib").exists():
                try:
                    shap_table, shap_scale = local_shap(frame)
                    shap_table["Direction"] = np.where(shap_table["SHAP contribution"] >= 0, "Raises raw risk", "Lowers raw risk")
                    fig = px.bar(shap_table.sort_values("SHAP contribution"), x="SHAP contribution", y="Feature", orientation="h",
                                 color="Direction", color_discrete_map={"Raises raw risk": "#efb264", "Lowers raw risk": "#58c4d9"}, title="Top 3 individual SHAP drivers")
                    st.plotly_chart(styled_chart(fig, 285), use_container_width=True)
                    st.caption(f"Contributions explain the underlying tree model's {shap_scale}. Positive values push its score upward. They do not sum to the displayed calibrated probability.")
                except (ValueError, TypeError, AttributeError, KeyError, ImportError) as error:
                    st.warning(f"The calibrated prediction is available; this model's local tree explanation could not be computed: {error}")

with tab_budget:
    st.subheader("Plan follow-up within a budget")
    st.caption("Scenario estimates use calibrated risk × assumed relative risk reduction. Intervention effects and avoided costs are editable assumptions, not measured causal effects.")
    predictions = csv_artifact("tables/test_predictions.csv")
    if predictions.empty:
        st.info("Finish model training to generate held-out patient probabilities for the simulator.")
    else:
        cohort = patient_allocation_cohort(predictions)
        st.caption(f"Allocation cohort: {len(cohort):,} unique held-out patients. The latest encounter per patient is retained using encounter ID as a time proxy ({len(predictions) - len(cohort):,} repeated encounters removed).")
        with st.form("budget_inputs"):
            budget = st.slider("Total follow-up budget ($)", 0, max(10000, len(cohort) * 100), int(np.floor(.1 * len(cohort)) * 50), step=50)
            columns = st.columns(3)
            costs, effects = [], []
            for i, (column, name, default_cost, default_effect) in enumerate(zip(columns, ["Phone call", "Home visit", "Case manager"], [50, 250, 600], [15, 25, 35])):
                with column:
                    st.markdown(f"**{name}**")
                    costs.append(st.slider("Cost per patient ($)", 10, 1500, default_cost, step=10, key=f"cost_{i}"))
                    effects.append(st.slider("Relative risk reduction (%)", 0, 80, default_effect, key=f"effect_{i}") / 100)
            a, b = st.columns(2)
            with a:
                readmission_cost = st.number_input("Assumed cost per readmission ($)", 0, 200000, 15000, step=500)
            with b:
                objective_label = st.selectbox("Optimization objective", ["Prevented readmissions", "Net savings"])
            run_simulation = st.form_submit_button("Optimize follow-up plan", type="primary", use_container_width=True)
        if run_simulation:
            with st.spinner("Solving the patient-level multiple-choice knapsack…"):
                optimized, budget_table = simulate_budget(cohort, budget, tuple(costs), tuple(effects), readmission_cost,
                    "prevented" if objective_label == "Prevented readmissions" else "net_savings")
            st.session_state["budget_result"] = (optimized, budget_table)
        if "budget_result" in st.session_state:
            optimized, budget_table = st.session_state.budget_result
            summary = optimized["summary"]
            st.caption("Showing the last submitted scenario. Submit again after editing inputs.")
        else:
            summary = json_artifact("artifacts/budget_metrics.json").get("optimized", {})
            budget_table = csv_artifact("tables/budget_comparison.csv")
            st.caption("Showing the saved default scenario. Submit the form to apply the current assumptions.")
        if summary:
            cards = st.columns(4)
            cards[0].metric("Patients selected", f"{summary['patients_selected']:,}")
            cards[1].metric("Expected prevented", f"{summary['expected_prevented_readmissions']:.2f}")
            cards[2].metric("Expected net savings", f"${summary['expected_net_savings']:,.0f}")
            cards[3].metric("Net ROI", f"{summary['roi']:.2f}×" if summary.get("roi") is not None else "—")
            gap = summary.get("mip_gap")
            st.caption(f"Budget ${summary['budget']:,.0f} · spent ${summary['spending']:,.0f} · {summary.get('solution_status', 'saved result')}" + (f" · relative solver gap {gap:.2%}" if gap is not None else ""))
        if not budget_table.empty:
            display_comparison(budget_table)
            st.caption("Random, first-come, top-risk, and lottery comparators use the cheapest tier at the same budget. Blanket coverage may exceed the budget and is explicitly marked infeasible. ROI = (avoided readmission cost − intervention spending) / spending.")
            fig = px.bar(budget_table, y="strategy", x="expected_prevented_readmissions", orientation="h", color="budget_feasible",
                          color_discrete_map={True: TEAL, False: "#a8866b"}, title="Expected benefit by allocation strategy")
            st.plotly_chart(styled_chart(fig, 400), use_container_width=True)
            st.download_button("Download this comparison", budget_table.to_csv(index=False), "readmitrisk_budget_comparison.csv", "text/csv")
        sensitivity = csv_artifact("tables/budget_sensitivity.csv")
        if not sensitivity.empty:
            with st.expander("Budget sensitivity under saved default assumptions"):
                fig = px.line(sensitivity, x="budget", y="expected_prevented_readmissions", markers=True,
                              hover_data=["solution_status"], title="Additional budget and feasible expected benefit")
                st.plotly_chart(styled_chart(fig), use_container_width=True)
                st.caption("Each point is a feasible solution. Time limits may lead to heuristic fallbacks; hover over each point for its solver status and consult the saved table for certified optimality.")

with tab_tiers:
    st.subheader("Patient profiles and care-team playbooks")
    cluster_metrics = json_artifact("artifacts/clustering.json")
    profiles = csv_artifact("tables/cluster_profiles.csv")
    scatter = csv_artifact("tables/cluster_scatter.csv")
    if not cluster_metrics or profiles.empty:
        st.info("Complete clustering in `python run_project.py` to display the saved patient tiers.")
    else:
        st.caption(f"K-Means selected {cluster_metrics['selected_k']} groups using training-only elbow, silhouette and Davies–Bouldin measures. Tiers are ordered by training readmission rates. The chart and rates below describe held-out encounters.")
        cards = st.columns(len(profiles))
        train_profiles = csv_artifact("tables/cluster_train_profiles.csv")
        tier_order = (train_profiles.sort_values("readmission_rate").tier.tolist() if not train_profiles.empty else profiles.tier.tolist())
        for card, tier in zip(cards, tier_order):
            row = profiles.loc[profiles.tier == tier].iloc[0]
            card.metric(tier, f"{row.readmission_rate:.1%}", help=f"{int(row.patients_or_encounters):,} held-out encounters")
        if not scatter.empty:
            fig = px.scatter(scatter, x="pc1", y="pc2", color="tier", render_mode="svg", category_orders={"tier": tier_order},
                             opacity=.5, hover_data=["readmit_30"], title="Held-out patient profiles in two dimensions")
            fig.update_traces(marker_size=5)
            variance = cluster_metrics.get("pca_explained_variance", [0, 0])
            fig.update_xaxes(title=f"PC1 ({variance[0]:.1%} of feature variance)")
            fig.update_yaxes(title=f"PC2 ({variance[1]:.1%} of feature variance)")
            st.plotly_chart(styled_chart(fig, 500), use_container_width=True)
            st.caption("PCA and scaling were fitted on training encounters. Up to 6,000 held-out encounters are shown; overlap is expected because clustering summarizes profiles rather than separating outcomes.")
        st.markdown("**Held-out tier profiles**")
        st.dataframe(profiles.drop(columns=["split", "cluster"], errors="ignore"), hide_index=True, use_container_width=True,
                     column_config={"readmission_rate": st.column_config.NumberColumn("Readmission rate", format="%.3f")})
        playbooks = csv_artifact("tables/cluster_playbooks.csv")
        if not playbooks.empty:
            st.markdown("**Suggested care-team playbooks**")
            st.dataframe(playbooks[["tier", "suggested_care_team_playbook"]], hide_index=True, use_container_width=True)
            st.caption("Playbooks are exploratory planning suggestions. High and Critical describe relative training-cohort risk, not a validated clinical triage scale.")
        with st.expander("Cluster selection and Gaussian mixture comparison"):
            st.dataframe(csv_artifact("tables/cluster_selection.csv"), hide_index=True, use_container_width=True)
            st.dataframe(csv_artifact("tables/cluster_algorithm_comparison.csv"), hide_index=True, use_container_width=True)
            st.caption(f"K-Means / Gaussian mixture adjusted Rand agreement: {cluster_metrics.get('gmm_adjusted_rand_agreement', 0):.3f}.")

with tab_metrics:
    st.subheader("Model evaluation results")
    st.caption("Metrics were computed on patient-isolated data. Accuracy is shown alongside recall, precision, PR-AUC, ROC-AUC, and calibration because readmission is an imbalanced outcome.")

    classifier_metrics = csv_artifact("tables/classifier_metrics.csv")
    model_comparison = csv_artifact("tables/model_comparison.csv")
    candidate_metrics = csv_artifact("tables/classifier_candidate_metrics.csv")
    sequence_metrics = csv_artifact("tables/sequence_model_comparison.csv")

    st.markdown("**Discharge-time classifier comparison**")
    if not model_comparison.empty:
        comparison_columns = [column for column in ["model", "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean", "cv_accuracy", "cv_recall", "full_fit_seconds"] if column in model_comparison]
        comparison_view = model_comparison[comparison_columns].sort_values("cv_pr_auc_mean", ascending=False)
        chart_columns = [column for column in ["cv_pr_auc_mean", "cv_roc_auc_mean"] if column in comparison_view]
        chart_metrics = comparison_view.melt(id_vars="model", value_vars=chart_columns, var_name="Metric", value_name="Score").dropna()
        chart_metrics["Metric"] = chart_metrics["Metric"].replace({"cv_pr_auc_mean": "Grouped CV PR-AUC", "cv_roc_auc_mean": "Grouped CV ROC-AUC"})
        fig = px.bar(chart_metrics, x="model", y="Score", color="Metric", barmode="group", text=chart_metrics["Score"].map(lambda value: f"{value:.3f}"),
                     title="Algorithm comparison on patient-grouped cross-validation")
        fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Score")
        fig.update_xaxes(title="")
        st.plotly_chart(styled_chart(fig, 390), use_container_width=True)
        comparison_display = metrics_display(comparison_view,
            percent_columns=["cv_accuracy", "cv_recall"],
            decimal_columns=["cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean"],
            seconds_columns=["full_fit_seconds"])
        comparison_display = comparison_display.rename(columns={"model": "Model", "cv_pr_auc_mean": "Grouped CV PR-AUC", "cv_pr_auc_std": "PR-AUC std.",
            "cv_roc_auc_mean": "CV ROC-AUC", "cv_accuracy": "Native CV accuracy", "cv_recall": "Native CV recall", "full_fit_seconds": "Fit time (s)"})
        st.dataframe(comparison_display, hide_index=True, use_container_width=True)
        st.caption("Tuned LightGBM was selected by grouped cross-validation PR-AUC. Native CV accuracy/recall use each model's default decision rule and are not the deployed threshold results.")
    else:
        st.info("Classifier-comparison results are unavailable.")

    st.markdown("**Four classifier families: validation-frozen accuracy-target evaluation**")
    if not candidate_metrics.empty:
        candidate_test = candidate_metrics.loc[candidate_metrics["split"].eq("test")].copy()
        candidate_columns = [column for column in ["model", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "threshold", "selection_rate"] if column in candidate_test]
        candidate_view = candidate_test[candidate_columns]
        candidate_chart = candidate_view.melt(id_vars="model", value_vars=["accuracy", "recall", "pr_auc"], var_name="Metric", value_name="Score")
        fig = px.bar(candidate_chart, x="model", y="Score", color="Metric", barmode="group", title="Held-out comparison with validation-frozen thresholds")
        fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Score")
        fig.update_xaxes(title="")
        st.plotly_chart(styled_chart(fig, 380), use_container_width=True)
        candidate_display = metrics_display(candidate_view, percent_columns=["accuracy", "balanced_accuracy", "precision", "recall", "f1", "threshold", "selection_rate"], decimal_columns=["roc_auc", "pr_auc", "brier"])
        st.dataframe(candidate_display.rename(columns={"model": "Model", "balanced_accuracy": "Balanced accuracy", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier score", "threshold": "Validation-frozen threshold", "selection_rate": "Patients alerted"}), hide_index=True, use_container_width=True)
        st.caption("Every listed threshold was selected on validation patients to satisfy an 82% validation-accuracy floor, then evaluated on new test patients. This exceeds the requested 70% accuracy target without resampling or changing the test distribution. Compare recall and precision as well as accuracy.")
    else:
        st.info("Candidate-family metrics are not in the currently saved artifacts. The algorithm comparison above is available; rerun the classifier phase with `--force` once to create this validation-frozen table.")

    st.markdown("**Selected calibrated LightGBM: independent held-out test results**")
    if not classifier_metrics.empty:
        final_view = classifier_metrics.loc[classifier_metrics["split"].eq("test")].copy()
        final_columns = [column for column in ["operating_point", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "threshold", "selection_rate"] if column in final_view]
        final_view = final_view[final_columns]
        final_view["operating_point"] = final_view["operating_point"].replace({"balanced": "Balanced / accuracy", "high_recall": "High recall"})
        operating_chart = final_view.melt(id_vars="operating_point", value_vars=["accuracy", "precision", "recall", "f1"], var_name="Metric", value_name="Rate")
        fig = px.bar(operating_chart, x="Metric", y="Rate", color="operating_point", barmode="group", text=operating_chart["Rate"].map(lambda value: f"{value:.1%}"),
                     title="Deployment operating-point trade-off on held-out patients")
        fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Rate")
        fig.update_xaxes(title="")
        st.plotly_chart(styled_chart(fig, 380), use_container_width=True)
        final_display = metrics_display(final_view, percent_columns=["accuracy", "balanced_accuracy", "precision", "recall", "f1", "threshold", "selection_rate"], decimal_columns=["roc_auc", "pr_auc", "brier"])
        st.dataframe(final_display.rename(columns={"operating_point": "Operating point", "balanced_accuracy": "Balanced accuracy", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier score", "threshold": "Risk threshold", "selection_rate": "Patients alerted"}), hide_index=True, use_container_width=True)
        st.caption("Balanced mode prioritizes maintaining accuracy; High recall mode identifies more recorded readmissions but creates more review alerts. Thresholds were selected on validation patients and frozen before testing.")
    else:
        st.info("Final classifier metrics are unavailable.")

    st.markdown("**Sequence-forecasting models: separate next-encounter task**")
    if not sequence_metrics.empty:
        sequence_test = sequence_metrics.loc[sequence_metrics["split"].eq("test")].copy()
        sequence_columns = [column for column in ["model", "operating_point", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"] if column in sequence_test]
        sequence_test = sequence_test[sequence_columns]
        sequence_test["operating_point"] = sequence_test["operating_point"].replace({"balanced": "Balanced / accuracy", "high_recall": "High recall"})
        sequence_display = metrics_display(sequence_test, percent_columns=["accuracy", "precision", "recall", "f1"], decimal_columns=["roc_auc", "pr_auc", "brier"])
        st.dataframe(sequence_display.rename(columns={"model": "Model", "operating_point": "Operating point", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier score"}), hide_index=True, use_container_width=True)
        st.caption("GRU, LSTM, and history-only XGBoost use earlier encounters to forecast the next recorded encounter. They use a repeated-encounter subset and therefore should not be compared directly with the discharge-time LightGBM metrics above.")
    else:
        st.info("Sequence-model results are unavailable.")

st.divider()
st.caption("ReadmitRisk · Nafiul Islam & Onika Tahmim Subha · Group 8 · Data Analytics Laboratory · Summer 2026")
