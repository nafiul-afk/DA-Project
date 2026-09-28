"""Run ReadmitRisk phases with transparent reuse of saved results.

Examples: python run_project.py --phase all
          python run_project.py --phase classifier --force
"""
from __future__ import annotations

import os
# Set thread budgets before NumPy/TensorFlow load on ordinary student laptops.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/readmitrisk-mpl")
os.environ.setdefault("PYTHONHASHSEED", "42")

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parent
PHASES = ["data", "eda", "classifier", "sequences", "allocation", "clustering", "report"]
RESULT_FILES = {
    "data": "data_quality.json", "eda": "eda.json", "classifier": "classifier_metrics.json",
    "sequences": "sequence_metrics.json", "allocation": "budget_metrics.json", "clustering": "clustering.json",
}


def load_prepared(root=ROOT):
    import pandas as pd
    import numpy as np
    return pd.read_csv(root / "data/cleaned_data.csv", keep_default_na=False), dict(np.load(root / "artifacts/split_indices.npz"))


def run_phase(phase, force=False, root=ROOT):
    root = Path(root)
    from readmitrisk.data import save_json
    if phase in RESULT_FILES and not force:
        path = root / "artifacts" / RESULT_FILES[phase]
        if path.exists():
            print(f"Using saved {phase} results. Pass --force to recompute.", flush=True)
            return json.loads(path.read_text())
    start = time.perf_counter()
    if phase == "data":
        from readmitrisk.data import prepare_data
        result = prepare_data(root)[2]
    elif phase in {"eda", "classifier", "sequences", "clustering"}:
        if not (root / "data/cleaned_data.csv").exists():
            run_phase("data", root=root)
        frame, splits = load_prepared(root)
        if phase == "eda":
            from readmitrisk.eda import run_eda
            result = run_eda(frame.iloc[splits["train"]], root)
        elif phase == "classifier":
            from readmitrisk.modeling import run_classifier
            result = run_classifier(frame, splits, root)
        elif phase == "sequences":
            from readmitrisk.sequences import run_sequences
            result = run_sequences(frame, splits, root)
        else:
            from readmitrisk.clustering import run_clustering
            result = run_clustering(frame, splits, root)
    elif phase == "allocation":
        import pandas as pd
        from readmitrisk.allocation import run_allocation
        if not (root / "tables/test_predictions.csv").exists():
            run_phase("classifier", root=root)
        result = run_allocation(pd.read_csv(root / "tables/test_predictions.csv"), root)
    elif phase == "report":
        from readmitrisk.reporting import build_presentation
        aggregate_metrics(root)
        result = str(build_presentation(root))
    else:
        raise ValueError(f"Unknown phase: {phase}")
    print(f"Completed {phase} in {time.perf_counter()-start:.1f}s.", flush=True)
    return result


def aggregate_metrics(root=ROOT):
    from readmitrisk.data import save_json
    names = dict(RESULT_FILES)
    names.update(feature_selection="feature_selection.json", fairness="fairness.json")
    aggregate = {key: json.loads((root / "artifacts" / file).read_text()) for key, file in names.items() if (root / "artifacts" / file).exists()}
    save_json(root / "artifacts/metrics.json", aggregate)
    files = [p for directory in ["readmitrisk", "data"] for p in (root / directory).glob("*") if p.is_file() and p.suffix in {".py", ".csv"}]
    provenance = dict(seed=42, python=platform.python_version(),
        sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        cache_policy="Saved phase results are explicitly reused by default. After changing data, code or assumptions, run --phase all --force to rebuild consistently.")
    save_json(root / "artifacts/provenance.json", provenance)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", *PHASES], default="all")
    parser.add_argument("--force", action="store_true", help="Recompute selected phases; use all after changing data or model code.")
    args = parser.parse_args()
    for phase in PHASES if args.phase == "all" else [args.phase]:
        run_phase(phase, force=args.force)
    aggregate_metrics()
    if (ROOT / "tables/classifier_metrics.csv").exists():
        import pandas as pd
        metrics = pd.read_csv(ROOT / "tables/classifier_metrics.csv")
        print("\nClassifier held-out metrics:")
        print(metrics.loc[metrics.split == "test", ["operating_point", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]].to_string(index=False))


if __name__ == "__main__":
    main()
