"""Train-only unsupervised patient segmentation and held-out tier profiles."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
CLUSTER_FEATURES = [
    "age_ordinal", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "med_changes_count", "meds_active_count", "lab_procs_per_day",
    "diag_diversity",
]


def _indices(split_indices, name):
    for key in (name, f"{name}_idx", f"{name}_indices"):
        if key in split_indices:
            return np.asarray(split_indices[key], dtype=int)
    raise KeyError(f"Missing {name} indices; supply positional train/test indices.")


def _tier_names(k):
    return {2: ["Low", "High"], 3: ["Low", "Moderate", "High"],
            4: ["Low", "Moderate", "High", "Critical"],
            5: ["Very low", "Low", "Moderate", "High", "Critical"],
            6: ["Very low", "Low", "Guarded", "Elevated", "High", "Critical"]}[k]


def _profile(frame, labels, tier_map, split):
    rows = []
    for cluster in sorted(tier_map):
        part = frame.iloc[np.flatnonzero(labels == cluster)]
        if not len(part):
            continue
        diagnosis = next((col for col in ("diag_1_group", "diag_1", "primary_diagnosis_group") if col in part), None)
        def average(col):
            return float(pd.to_numeric(part[col], errors="coerce").mean()) if col in part else None
        rows.append({
            "split": split, "cluster": int(cluster), "tier": tier_map[cluster], "patients_or_encounters": int(len(part)),
            "mean_age": average("age_ordinal"), "mean_los": average("time_in_hospital"),
            "mean_prior_visits": average("total_prior_visits"), "mean_medications": average("num_medications"),
            "dominant_diagnosis": str(part[diagnosis].mode().iloc[0]) if diagnosis and not part[diagnosis].mode().empty else "Unknown",
            "readmission_rate": average("readmit_30"),
        })
    return pd.DataFrame(rows)


def run_clustering(df, split_indices, output_dir="."):
    """Fit k=2..6 on training data; use outcomes only to name the fitted tiers.

    Cluster count is chosen by the mean rank of three internal training-only
    measures: silhouette, Davies–Bouldin, and normalized elbow distance. The
    held-out outcome is never used to fit clusters or order tier names.
    """
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    for folder in ("artifacts", "tables", "figures"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    train_idx, test_idx = _indices(split_indices, "train"), _indices(split_indices, "test")
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    if "patient_nbr" in df and set(train.patient_nbr) & set(test.patient_nbr):
        raise ValueError("Clustering train and test patients overlap.")
    features = [col for col in CLUSTER_FEATURES if col in df and pd.to_numeric(train[col], errors="coerce").notna().any()]
    if len(features) < 3 or len(train) < 10 or not len(test):
        raise ValueError("Clustering requires three numeric features, ten train rows, and a held-out set.")
    prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    x_train = prep.fit_transform(train[features].apply(pd.to_numeric, errors="coerce"))
    x_test = prep.transform(test[features].apply(pd.to_numeric, errors="coerce"))
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(x_train), min(1800, len(x_train)), replace=False)
    rows, fitted = [], {}
    with threadpool_limits(limits=2):
        for k in range(2, 7):
            estimator = KMeans(n_clusters=k, n_init=10, random_state=SEED, max_iter=200)
            labels = estimator.fit_predict(x_train)
            sample_labels = labels[sample]
            sil = silhouette_score(x_train[sample], sample_labels) if len(np.unique(sample_labels)) > 1 else -1
            rows.append({"k": k, "inertia": float(estimator.inertia_), "silhouette": float(sil),
                         "davies_bouldin": float(davies_bouldin_score(x_train, labels))})
            fitted[k] = estimator
        scores = pd.DataFrame(rows)
        normalized_k = (scores.k - scores.k.min()) / (scores.k.max() - scores.k.min())
        inertia_range = scores.inertia.max() - scores.inertia.min()
        normalized_inertia = (scores.inertia - scores.inertia.min()) / max(inertia_range, 1e-12)
        scores["elbow_distance"] = np.maximum(0, (1 - normalized_k - normalized_inertia) / np.sqrt(2))
        scores["mean_metric_rank"] = pd.concat([
            scores.silhouette.rank(ascending=False), scores.davies_bouldin.rank(),
            scores.elbow_distance.rank(ascending=False)], axis=1).mean(axis=1)
        selected_k = int(scores.sort_values(["mean_metric_rank", "k"]).iloc[0].k)
        kmeans = fitted[selected_k]
        train_labels = kmeans.labels_
        test_labels = kmeans.predict(x_test)
        gmm = GaussianMixture(n_components=selected_k, covariance_type="full", random_state=SEED,
                              n_init=2, max_iter=150, reg_covar=1e-5)
        gmm.fit(x_train)
        gmm_labels = gmm.predict(x_train)
        gmm_sample = gmm_labels[sample]
        gmm_silhouette = float(silhouette_score(x_train[sample], gmm_sample)) if len(np.unique(gmm_sample)) > 1 else None
        gmm_db = float(davies_bouldin_score(x_train, gmm_labels)) if len(np.unique(gmm_labels)) > 1 else None
        pca = PCA(n_components=2, random_state=SEED).fit(x_train)
        projection = pca.transform(x_test)
    rates = pd.DataFrame({"cluster": train_labels, "readmit_30": train.readmit_30.to_numpy()}).groupby("cluster").readmit_30.mean().sort_values()
    names = _tier_names(selected_k)
    tier_map = {int(cluster): names[rank] for rank, cluster in enumerate(rates.index)}
    train_profiles = _profile(train, train_labels, tier_map, "Train")
    test_profiles = _profile(test, test_labels, tier_map, "Held-out test")
    playbook_rows = []
    for rank, cluster in enumerate(rates.index):
        proportion = rank / max(selected_k - 1, 1)
        if proportion < .25:
            care = "Standard discharge education; verify medication list and routine follow-up contact."
        elif proportion < .55:
            care = "Early phone check; confirm follow-up attendance, glucose-monitoring access, and medication understanding."
        elif proportion < .8:
            care = "Nurse follow-up within the locally agreed interval; review prior admissions and coordinate outpatient support."
        else:
            care = "Care-team review for intensive case management; assess social barriers and coordinate a personalized transition plan."
        playbook_rows.append({"cluster": int(cluster), "tier": names[rank],
                              "training_readmission_rate": float(rates.loc[cluster]),
                              "suggested_care_team_playbook": care})
    playbooks = pd.DataFrame(playbook_rows)
    scatter = pd.DataFrame({"pc1": projection[:, 0], "pc2": projection[:, 1], "cluster": test_labels,
                            "tier": [tier_map[int(label)] for label in test_labels],
                            "readmit_30": test.readmit_30.to_numpy()})
    if "patient_nbr" in test:
        scatter["patient_nbr"] = test.patient_nbr.to_numpy()
    if len(scatter) > 6000:
        scatter = scatter.sample(6000, random_state=SEED)
    scores.to_csv(output / "tables/cluster_selection.csv", index=False)
    train_profiles.to_csv(output / "tables/cluster_train_profiles.csv", index=False)
    test_profiles.to_csv(output / "tables/cluster_profiles.csv", index=False)
    playbooks.to_csv(output / "tables/cluster_playbooks.csv", index=False)
    scatter.to_csv(output / "tables/cluster_scatter.csv", index=False)
    comparison = pd.DataFrame([
        {"algorithm": "K-Means", "k": selected_k, "silhouette": float(scores.loc[scores.k == selected_k, "silhouette"].iloc[0]),
         "davies_bouldin": float(scores.loc[scores.k == selected_k, "davies_bouldin"].iloc[0]), "converged": bool(kmeans.n_iter_ < kmeans.max_iter)},
        {"algorithm": "GaussianMixture", "k": selected_k, "silhouette": gmm_silhouette,
         "davies_bouldin": gmm_db, "converged": bool(gmm.converged_)},
    ])
    comparison.to_csv(output / "tables/cluster_algorithm_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, measure, title in zip(axes, ("inertia", "silhouette", "davies_bouldin"),
                                  ("Elbow: lower inertia", "Silhouette: higher is better", "Davies–Bouldin: lower is better")):
        ax.plot(scores.k, scores[measure], "o-", color="#0b9f96")
        ax.axvline(selected_k, color="#d48836", linestyle="--", alpha=.7)
        ax.set(xlabel="Number of clusters", title=title, xticks=scores.k)
        ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(output / "figures/cluster_selection.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap("viridis")(np.linspace(.15, .9, selected_k))
    for name, color in zip(names, colors):
        subset = scatter[scatter.tier == name]
        ax.scatter(subset.pc1, subset.pc2, s=8, alpha=.42, color=color, label=name)
    ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
           title="Held-out patient encounter tiers (PCA fitted on training data)")
    ax.legend(title="Training-ordered tier", frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(output / "figures/cluster_pca.png", dpi=180)
    plt.close(fig)
    joblib.dump({"preprocessor": prep, "model": kmeans, "gmm": gmm, "pca": pca,
                 "features": features, "cluster_to_tier": tier_map}, output / "artifacts/clustering.joblib")
    metrics = {
        "selected_k": selected_k, "features": features, "train_rows": len(train), "test_rows": len(test),
        "selection_rule": "Lowest mean rank across training silhouette (high), Davies–Bouldin (low), and normalized elbow distance (high); ties prefer smaller k.",
        "tier_naming": "Names ordered by TRAINING observed readmission rates after unsupervised fitting; outcomes and patient identifiers are excluded from the cluster features.",
        "unit": "Encounter-level profiles with disjoint train/test patients; repeat encounters remain within their patient's split.",
        "care_note": "Exploratory segmentation and suggested playbooks, requiring prospective care-team validation.",
        "cluster_to_tier": tier_map, "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "gmm_adjusted_rand_agreement": float(adjusted_rand_score(train_labels, gmm_labels)),
        "algorithm_comparison": comparison.replace({np.nan: None}).to_dict("records"),
        "selection_scores": scores.to_dict("records"),
        "train_profiles": train_profiles.replace({np.nan: None}).to_dict("records"),
        "test_profiles": test_profiles.replace({np.nan: None}).to_dict("records"),
        "playbooks": playbooks.to_dict("records"),
    }
    with open(output / "artifacts/clustering.json", "w") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    return metrics
