"""Development-only exploratory analysis, with patient-independent inference.

Charts describe encounters. Hypothesis tests use the first recorded eligible
encounter per patient, so a frequently admitted person is not counted as many
independent observations. These are associations, not treatment effects.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu

TEAL = "#0F8B8D"
DARK = "#153C47"
MINT = "#78C9B5"
ORANGE = "#F1A45B"


def _holm(pvalues: list[float]) -> list[float]:
    """Holm family-wise adjustment: protect against five hypothesis tests."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, previous)
    return adjusted.tolist()


def _rate_table(df: pd.DataFrame, column: str, order=None) -> pd.DataFrame:
    table = df.groupby(column, observed=True, dropna=False)["readmit_30"].agg(
        encounters="size", readmissions="sum", rate="mean"
    ).reset_index()
    table[column] = table[column].astype(str)
    if order is not None:
        positions = {str(value): i for i, value in enumerate(order)}
        table = table.assign(_order=table[column].map(positions).fillna(len(positions))).sort_values("_order").drop(columns="_order")
    else:
        table = table.sort_values(["rate", "encounters"], ascending=[False, False])
    return table.reset_index(drop=True)


def run_eda(df: pd.DataFrame, output_dir: Path | str = Path("."), cohort_label: str = "Training partition") -> dict:
    """Save charts/tables, then return a JSON-compatible summary of real results."""
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    artifacts = output_dir / "artifacts"
    for folder in (figures, tables, artifacts):
        folder.mkdir(parents=True, exist_ok=True)
    data = df.copy()
    data["prior_inpatient_band"] = pd.cut(
        data["number_inpatient"], bins=[-1, 0, 1, 2, 4, np.inf], labels=["0", "1", "2", "3–4", "5+"]
    )
    data["medication_count_band"] = pd.cut(
        data["num_medications"], bins=[-1, 5, 10, 15, 20, 30, np.inf],
        labels=["0–5", "6–10", "11–15", "16–20", "21–30", "31+"]
    )
    data["segment_age"] = pd.cut(
        data["age_ordinal"], bins=[-1, 40, 60, 80, 110], right=False,
        labels=["Under 40", "40–59", "60–79", "80+"]
    )
    data["segment_prior"] = pd.cut(data["number_inpatient"], bins=[-1, 0, 1, np.inf], labels=["0 prior", "1 prior", "2+ prior"])
    for col in ["race", "gender", "age", "admission_type", "discharge_disposition", "admission_source", "diag_1_group", "A1Cresult", "insulin", "change", "diabetesMed"]:
        if col in data:
            data[col] = data[col].astype("object").fillna("Unknown")
    # Administrative labels are lengthy and several occur only a handful of
    # times. Pool <1% levels using this training partition alone, while saving
    # the original counts for transparent audit. This is descriptive pooling;
    # predictive pipelines independently learn their own fold-specific pools.
    display_names = {
        "Discharged/transferred to another rehab fac including rehab units of a hospital .": "Rehabilitation facility",
        "Discharged/transferred to another type of inpatient care institution": "Other inpatient institution",
        "Discharged/transferred to another short term hospital": "Another short-term hospital",
        "Discharged/transferred to SNF": "Skilled nursing facility",
        "Discharged/transferred to home with home health service": "Home with health service",
        "Discharged to home": "Home",
        "Transfer from another health care facility": "Another healthcare facility",
        "Transfer from a hospital": "Hospital transfer",
    }
    for col in ["admission_type", "discharge_disposition", "admission_source"]:
        _rate_table(data, col).to_csv(tables / f"eda_{col}_unpooled.csv", index=False)
        frequency = data[col].value_counts(normalize=True)
        common = set(frequency[frequency >= .01].index) | {"Unknown"}
        data[col] = data[col].where(data[col].isin(common), "Other (rare levels)").replace(display_names)
    base_rate = float(data["readmit_30"].mean())
    sns.set_theme(style="whitegrid", font_scale=1.02, rc={
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "text.color": DARK, "axes.labelcolor": DARK,
        "grid.alpha": 0.25, "figure.facecolor": "white", "axes.facecolor": "white",
    })
    captions = []
    rates: dict[str, pd.DataFrame] = {}

    def finish(fig, slug: str, title: str, insight: str):
        fig.suptitle(title, x=0.06, y=0.98, ha="left", fontsize=17, color=DARK, fontweight="bold")
        fig.text(0.06, 0.04, textwrap.fill(insight, width=120), fontsize=10.5, color=DARK, va="bottom")
        fig.text(0.06, 0.01, f"{cohort_label} only • encounter-level descriptive associations • ReadmitRisk / Group 8", fontsize=8, color="#61757B")
        fig.tight_layout(rect=[0.025, 0.15, 0.99, 0.92])
        path = figures / f"eda_{slug}.png"
        fig.savefig(path, dpi=175, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        captions.append({"figure": str(path.relative_to(output_dir)), "title": title, "insight": insight})

    def plot_rate(col: str, title: str, order=None):
        table = _rate_table(data, col, order)
        rates[col] = table
        table.to_csv(tables / f"eda_{col}.csv", index=False)
        # Small groups stay visible but are excluded from the headline ranking.
        eligible = table[table["encounters"] >= 100]
        if eligible.empty:
            eligible = table
        highest = eligible.iloc[int(np.argmax(eligible["rate"].values))]
        lowest = eligible.iloc[int(np.argmin(eligible["rate"].values))]
        insight = (f"{highest[col]}: {highest['rate']:.1%} readmitted ({int(highest['readmissions']):,}/{int(highest['encounters']):,}); "
                   f"{lowest[col]}: {lowest['rate']:.1%}. These are descriptive differences, not causes.")
        fig, ax = plt.subplots(figsize=(11.6, max(5.6, len(table) * 0.46 + 2.1)))
        bars = ax.barh(np.arange(len(table)), table["rate"], color=TEAL)
        ax.set_yticks(np.arange(len(table)), [textwrap.fill(str(x), 36) for x in table[col]])
        ax.invert_yaxis()
        ax.axvline(base_rate, ls="--", lw=1.5, color=ORANGE, label=f"Overall {base_rate:.1%}")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("30-day readmission rate")
        ax.set_ylabel("")
        max_rate = float(table["rate"].max())
        ax.set_xlim(0, max(max_rate * 1.42, 0.08))
        for bar, (_, row) in zip(bars, table.iterrows()):
            ax.text(bar.get_width() + max(max_rate * 0.025, 0.001), bar.get_y() + bar.get_height() / 2,
                    f"{row['rate']:.1%}  ·  n={int(row['encounters']):,}", va="center", fontsize=9)
        ax.legend(loc="lower right", frameon=False)
        finish(fig, col, title, insight)
        return table

    age_order = data.groupby("age", observed=True)["age_ordinal"].mean().sort_values().index.tolist()
    specs = [
        ("age", "Age: risk is not uniform across life stages", age_order),
        ("admission_type", "Admission type and 30-day readmission", None),
        ("discharge_disposition", "Discharge destination identifies follow-up needs", None),
        ("admission_source", "Admission source and 30-day readmission", None),
        ("diag_1_group", "Primary diagnosis: different groups, different needs", None),
        ("race", "Race: descriptive differences need a fairness audit", None),
        ("gender", "Gender: compare rates without assuming causation", None),
        ("prior_inpatient_band", "Prior inpatient visits flag repeat-use patterns", ["0", "1", "2", "3–4", "5+"]),
        ("insulin", "Insulin status and observed readmission", ["No", "Steady", "Up", "Down"]),
        ("change", "Medication change is a marker of care complexity", ["No", "Ch"]),
        ("diabetesMed", "Diabetes medication use and observed readmission", ["No", "Yes"]),
        ("A1Cresult", "A1C testing status and readmission", ["Not tested", "Norm", ">7", ">8"]),
        ("medication_count_band", "Medication burden and follow-up planning", ["0–5", "6–10", "11–15", "16–20", "21–30", "31+"]),
    ]
    for column, title, order in specs:
        if column in data.columns:
            plot_rate(column, title, order)

    # Length of stay: proportion within each class makes unequal classes comparable.
    los_stats = data.groupby("readmit_30")["time_in_hospital"].agg(["count", "mean", "median", "std"])
    los_stats.to_csv(tables / "eda_length_of_stay.csv")
    fig, ax = plt.subplots(figsize=(11.6, 6))
    for label, color, status in [("No <30-day readmission", MINT, 0), ("Readmitted <30 days", TEAL, 1)]:
        vals = data.loc[data["readmit_30"] == status, "time_in_hospital"]
        hist = vals.value_counts(normalize=True).sort_index()
        ax.plot(hist.index, hist.values, marker="o", color=color, lw=2.5, label=label)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Days in hospital", ylabel="Share within outcome group")
    ax.legend(frameon=False)
    los_insight = (f"Mean stay was {los_stats.loc[1, 'mean']:.2f} days for readmitted encounters versus {los_stats.loc[0, 'mean']:.2f} days for others; "
                   "a distribution test below checks the association on independent patients.")
    finish(fig, "length_of_stay", "Length of stay: compare within each outcome group", los_insight)

    numeric = [c for c in ["age_ordinal", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
                          "number_emergency", "number_inpatient", "number_diagnoses", "total_prior_visits", "med_changes_count", "meds_active_count",
                          "lab_procs_per_day", "diag_diversity", "polypharmacy", "readmit_30"] if c in data]
    corr = data[numeric].corr(method="spearman")
    corr.to_csv(tables / "eda_numeric_spearman.csv")
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool), k=1), cmap=sns.diverging_palette(20, 180, as_cmap=True), center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"fontsize": 7}, linewidths=0.5, cbar_kws={"label": "Spearman correlation", "shrink": 0.7}, ax=ax)
    ax.set_xticklabels([x.replace("_", " ") for x in corr.columns], rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels([x.replace("_", " ") for x in corr.index], fontsize=8)
    top_corr = corr["readmit_30"].drop("readmit_30").dropna().abs().idxmax()
    corr_insight = f"{top_corr.replace('_', ' ')} has the strongest numeric rank association with readmission (ρ={corr.loc[top_corr, 'readmit_30']:.3f}); no single numeric field explains risk."
    finish(fig, "numeric_correlation", "Numeric features: related signals and redundancy", corr_insight)

    segments = data.groupby(["segment_age", "diag_1_group", "segment_prior"], observed=True)["readmit_30"].agg(
        encounters="size", readmissions="sum", rate="mean").reset_index()
    segments = segments[segments["encounters"] >= 100].sort_values(["rate", "encounters"], ascending=False)
    segments["lift_vs_overall"] = segments["rate"] / base_rate
    segments.to_csv(tables / "eda_risk_segments_all.csv", index=False)
    top_segments = segments.head(10).copy()
    top_segments.to_csv(tables / "eda_top10_risk_segments.csv", index=False)
    labels = top_segments.apply(lambda r: f"{r['segment_age']} · {r['diag_1_group']} · {r['segment_prior']}", axis=1)
    fig, ax = plt.subplots(figsize=(12.2, 7.2))
    ax.barh(range(len(top_segments)), top_segments["rate"], color=TEAL)
    ax.set_yticks(range(len(top_segments)), labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("30-day readmission rate (segments with at least 100 encounters)")
    ax.axvline(base_rate, color=ORANGE, ls="--", label=f"Overall {base_rate:.1%}")
    ax.set_xlim(0, max(0.1, float(top_segments["rate"].max()) * 1.45))
    for i, (_, row) in enumerate(top_segments.iterrows()):
        ax.text(row["rate"] + 0.005, i, f"{row['rate']:.1%} ({int(row['readmissions'])}/{int(row['encounters'])})", va="center", fontsize=9)
    ax.legend(loc="lower right", frameon=False)
    best = top_segments.iloc[0]
    segment_insight = (f"Highest observed segment: {best['segment_age']} / {best['diag_1_group']} / {best['segment_prior']}: "
                       f"{best['rate']:.1%} ({int(best['readmissions'])}/{int(best['encounters'])}), {best['lift_vs_overall']:.1f}× overall. Exploratory ranking needs prospective validation.")
    finish(fig, "top10_risk_segments", "Top risk segments: prioritize sufficiently large groups", segment_insight)

    # Keep one earliest eligible development encounter per patient for inference.
    first = data.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first").copy()
    hypotheses = []

    def chi_test(key: str, hypothesis: str, column: str, expected_direction: bool = True):
        tab = pd.crosstab(first[column], first["readmit_30"])
        stat, p, dof, expected = chi2_contingency(tab)
        hypotheses.append({
            "id": key, "hypothesis": hypothesis, "test": "Pearson chi-square", "feature": column,
            "statistic": float(stat), "p_value": float(p), "degrees_of_freedom": int(dof),
            "n_patients": int(tab.to_numpy().sum()), "effect_size_name": "Cramer's V",
            "effect_size": float(np.sqrt(stat / (tab.to_numpy().sum() * min(tab.shape[0] - 1, tab.shape[1] - 1)))),
            "minimum_expected_count": float(expected.min()), "fraction_expected_below_5": float((expected < 5).mean()),
            "direction_consistent": bool(expected_direction),
        })

    p0 = float(first.loc[first["number_inpatient"] == 0, "readmit_30"].mean())
    p2 = float(first.loc[first["number_inpatient"] >= 2, "readmit_30"].mean())
    chi_test("H1", "More prior inpatient visits are associated with higher 30-day readmission risk.", "prior_inpatient_band", p2 > p0)
    hypotheses[-1]["direction_evidence"] = f"0 prior: {p0:.2%}; 2+ prior: {p2:.2%} (first encounter per patient)."
    pos = first.loc[first["readmit_30"] == 1, "time_in_hospital"].dropna()
    neg = first.loc[first["readmit_30"] == 0, "time_in_hospital"].dropna()
    u, p = mannwhitneyu(pos, neg, alternative="greater", method="asymptotic")
    hypotheses.append({
        "id": "H2", "hypothesis": "Readmitted patients tend to have longer index stays.", "test": "One-sided Mann–Whitney U", "feature": "time_in_hospital",
        "statistic": float(u), "p_value": float(p), "n_patients": int(len(pos) + len(neg)), "effect_size_name": "Rank-biserial correlation",
        "effect_size": float(2 * u / (len(pos) * len(neg)) - 1), "direction_consistent": bool(pos.mean() > neg.mean()),
        "direction_evidence": f"Mean/median days: readmitted {pos.mean():.2f}/{pos.median():.0f}; others {neg.mean():.2f}/{neg.median():.0f}.",
    })
    for column, suffix in [("insulin", "insulin"), ("change", "change"), ("diabetesMed", "diabetes_med")]:
        chi_test(f"H3_{suffix}", f"Readmission frequency differs across {column} categories.", column)
    adjusted = _holm([h["p_value"] for h in hypotheses])
    for result, corrected in zip(hypotheses, adjusted):
        result["holm_adjusted_p"] = float(corrected)
        result["conclusion"] = "Supported" if corrected < 0.05 and result["direction_consistent"] else "Not supported"
        result["interpretation"] = "Evidence of association only; observational data do not establish an intervention effect."
    pd.DataFrame(hypotheses).to_csv(tables / "eda_hypothesis_tests.csv", index=False)

    def highest_rate(column: str):
        table = rates[column]
        return table.loc[table["encounters"] >= 100].sort_values("rate", ascending=False).iloc[0]

    prior = rates["prior_inpatient_band"].set_index("prior_inpatient_band")
    age = highest_rate("age")
    diag = highest_rate("diag_1_group")
    discharge = highest_rate("discharge_disposition")
    insulin = highest_rate("insulin")
    a1c = highest_rate("A1Cresult")
    meds = highest_rate("medication_count_band")
    business_insights = [
        f"The {cohort_label.lower()} contains {len(data):,} encounters from {data['patient_nbr'].nunique():,} patients; {int(data['readmit_30'].sum()):,} ({base_rate:.2%}) were readmitted within 30 days. Always predicting no readmission would be {1-base_rate:.2%} accurate, so accuracy alone is misleading.",
        f"Prior inpatient burden is useful for triage: no prior visits had {prior.loc['0', 'rate']:.1%} readmission ({int(prior.loc['0', 'encounters']):,} encounters), while 5+ had {prior.loc['5+', 'rate']:.1%} ({int(prior.loc['5+', 'encounters']):,}); H1 adjusted p={hypotheses[0]['holm_adjusted_p']:.3g}.",
        los_insight,
        f"Among age groups with at least 100 encounters, {age['age']} had the highest observed rate, {age['rate']:.1%} ({int(age['readmissions']):,}/{int(age['encounters']):,}); age alone should not decide access to care.",
        f"{diag['diag_1_group']} was the highest-rate primary diagnosis group at {diag['rate']:.1%} ({int(diag['readmissions']):,}/{int(diag['encounters']):,}), helping define a follow-up workflow.",
        f"Discharge to {discharge['discharge_disposition']} had {discharge['rate']:.1%} readmission ({int(discharge['readmissions']):,}/{int(discharge['encounters']):,}); transitions of care deserve review.",
        f"Insulin status '{insulin['insulin']}' had {insulin['rate']:.1%} readmission ({int(insulin['encounters']):,} encounters). This may reflect illness severity; it is not evidence that insulin causes readmission.",
        f"A1C category '{a1c['A1Cresult']}' had {a1c['rate']:.1%} readmission ({int(a1c['encounters']):,} encounters); testing patterns and patient mix can confound this comparison.",
        f"The {meds['medication_count_band']}-medication group had the highest observed medication-band rate, {meds['rate']:.1%} ({int(meds['encounters']):,} encounters), suggesting medication review as an operational hypothesis.",
        segment_insight,
    ]
    summary = {
        "cohort": f"{cohort_label} only; held-out patients never enter EDA.",
        "encounters": int(len(data)), "patients": int(data["patient_nbr"].nunique()), "readmissions": int(data["readmit_30"].sum()),
        "readmission_rate": base_rate, "mean_length_of_stay": float(data["time_in_hospital"].mean()),
        "figures": captions, "chart_count": len(captions), "hypotheses": hypotheses,
        "inference_design": f"First eligible encounter ordered by encounter_id for each of {len(first):,} development patients; Holm correction for five prespecified tests at family-wise alpha 0.05. Encounter ID is only an ordering proxy. H1's chi-square is an omnibus test; its stated direction is checked using 2+ versus zero prior admissions, not assumed to prove a monotonic dose response.",
        "descriptive_caution": "Chart rates describe encounters, including repeat encounters. Administrative levels below 1% of training encounters are pooled; unpooled counts are also saved. All risk segments are exploratory with at least 100 encounters; their sample size is not an independent-patient count.",
        "key_business_insights": business_insights,
        "top_risk_segments": json.loads(top_segments.to_json(orient="records")),
    }
    (artifacts / "eda.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(captions).to_csv(tables / "eda_figure_captions.csv", index=False)
    (tables / "eda_business_insights.md").write_text("\n".join(f"- {line}" for line in business_insights) + "\n", encoding="utf-8")
    print(f"Phase 2 complete: {len(captions)} development-only charts; {len(first):,} independent patients for hypothesis tests.")
    for h in hypotheses:
        print(f"  {h['id']}: {h['conclusion']}; Holm-adjusted p={h['holm_adjusted_p']:.4g}; effect={h['effect_size']:.4f}")
    return summary
