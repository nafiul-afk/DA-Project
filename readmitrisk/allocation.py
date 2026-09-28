"""Patient-level, budget-constrained follow-up allocation.

All benefits in this module are scenario estimates: intervention effectiveness
is supplied by the user, not learned from this observational data set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

SEED = 42
TIER_NAMES = ("Phone call", "Home visit", "Case manager")


def _validate(probabilities, budget, costs, effectiveness, readmission_cost):
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    costs = np.asarray(costs, dtype=float).reshape(-1)
    effects = np.asarray(effectiveness, dtype=float).reshape(-1)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite numbers between 0 and 1.")
    if len(costs) != len(effects) or not len(costs):
        raise ValueError("Supply one effectiveness assumption for each intervention cost.")
    if not np.all(np.isfinite(costs)) or np.any(costs <= 0):
        raise ValueError("Intervention costs must be positive and finite.")
    if not np.all(np.isfinite(effects)) or np.any((effects < 0) | (effects > 1)):
        raise ValueError("Effectiveness must be a relative risk reduction between 0 and 1.")
    if not np.isfinite(budget) or budget < 0 or not np.isfinite(readmission_cost) or readmission_cost < 0:
        raise ValueError("Budget and readmission cost must be finite and nonnegative.")
    return p, costs, effects


def _assignments(p, choices, costs, effects, readmission_cost):
    chosen = choices >= 0
    spend = np.zeros(len(p))
    reduction = np.zeros(len(p))
    spend[chosen] = costs[choices[chosen]]
    reduction[chosen] = effects[choices[chosen]]
    names = [TIER_NAMES[i] if i < len(TIER_NAMES) else f"Tier {i + 1}" for i in range(len(costs))]
    return pd.DataFrame({
        "patient_position": np.arange(len(p)), "probability": p,
        "tier": choices, "intervention": [names[i] if i >= 0 else "No intervention" for i in choices],
        "selected": chosen, "cost": spend, "effectiveness": reduction,
        "expected_prevented": p * reduction,
        "expected_avoided_cost": p * reduction * readmission_cost,
        "expected_net_savings": p * reduction * readmission_cost - spend,
    })


def summarize_assignments(assignments, budget):
    """ROI is net savings / intervention spending; zero spending has no ROI."""
    spending = float(assignments["cost"].sum())
    prevented = float(assignments["expected_prevented"].sum())
    avoided = float(assignments["expected_avoided_cost"].sum())
    return {
        "patients_selected": int(assignments["selected"].sum()),
        "spending": spending, "budget": float(budget),
        "budget_feasible": bool(spending <= budget + 1e-6),
        "expected_prevented_readmissions": prevented,
        "expected_avoided_cost": avoided, "expected_net_savings": avoided - spending,
        "roi": (avoided - spending) / spending if spending else None,
    }


def allocate_budget(probabilities, budget, costs=(50, 250, 600),
                    effectiveness=(.15, .25, .35), readmission_cost=15000,
                    objective="prevented", time_limit=15.0, mip_rel_gap=.005):
    """Solve a genuine multiple-choice binary knapsack with SciPy/HiGHS.

    One variable represents each patient/intervention pair. Per-patient sums
    are at most one, so a patient can never consume two intervention budgets.
    A time-limited feasible solution is explicitly distinguished from a proven
    optimum. A deterministic feasible heuristic is retained if no valid solver
    incumbent exists (or if the solver's incumbent is weaker).
    """
    p, costs, effects = _validate(probabilities, budget, costs, effectiveness, readmission_cost)
    if objective not in {"prevented", "net_savings"}:
        raise ValueError("objective must be 'prevented' or 'net_savings'.")
    n, tiers = len(p), len(costs)
    values = p[:, None] * effects[None, :]
    if objective == "net_savings":
        values = values * readmission_cost - costs[None, :]
    choices = np.full(n, -1, dtype=int)
    remaining = float(budget)
    # A feasible starting comparison; this is never called an optimum.
    efficiency = values / costs[None, :]
    for index in np.argsort(-efficiency.ravel(), kind="stable"):
        patient, tier = divmod(int(index), tiers)
        if choices[patient] < 0 and values[patient, tier] > 0 and costs[tier] <= remaining + 1e-8:
            choices[patient] = tier
            remaining -= costs[tier]
    summary_solver = {
        "solver": "scipy.optimize.milp / HiGHS", "objective": objective,
        "solution_status": "feasible heuristic", "proven_optimal": False,
        "solver_status_code": None, "solver_message": "No feasible positive-value intervention.",
        "mip_gap": None, "time_limit_seconds": float(time_limit),
    }
    if n and budget >= costs.min() and np.any(values > 0):
        variables = np.arange(n * tiers)
        matrix = coo_matrix((
            np.r_[np.ones(n * tiers), np.tile(costs, n)],
            (np.r_[np.repeat(np.arange(n), tiers), np.full(n * tiers, n)],
             np.r_[variables, variables])), shape=(n + 1, n * tiers)).tocsc()
        constraint = LinearConstraint(matrix, np.zeros(n + 1), np.r_[np.ones(n), budget])
        result = milp(c=-values.ravel(), integrality=np.ones(n * tiers),
                      bounds=Bounds(0, 1), constraints=constraint,
                      options={"time_limit": max(.01, float(time_limit)), "mip_rel_gap": float(mip_rel_gap)})
        summary_solver.update(solver_status_code=int(result.status), solver_message=str(result.message))
        incumbent_valid = result.x is not None and np.all(np.isfinite(result.x))
        if incumbent_valid:
            binary = np.rint(result.x).astype(int).reshape(n, tiers)
            incumbent_valid = (np.max(np.abs(result.x - binary.ravel())) < 1e-4
                               and np.all((binary >= 0) & (binary <= 1))
                               and np.all(binary.sum(axis=1) <= 1)
                               and float((binary * costs).sum()) <= budget + 1e-6)
            heuristic_value = float(sum(values[i, t] for i, t in enumerate(choices) if t >= 0))
            if incumbent_valid and float((binary * values).sum()) >= heuristic_value - 1e-8:
                choices = np.where(binary.sum(axis=1) > 0, binary.argmax(axis=1), -1)
                gap = getattr(result, "mip_gap", None)
                gap = float(gap) if gap is not None and np.isfinite(gap) else None
                summary_solver.update(
                    solution_status="optimal within solver tolerance" if result.status == 0 else "time-limited feasible MILP",
                    proven_optimal=bool(result.status == 0 and gap is not None and gap <= 1e-9),
                    mip_gap=gap,
                )
    else:
        summary_solver.update(solution_status="optimal: no positive-value affordable choice", proven_optimal=True, mip_gap=0.0)
    assignments = _assignments(p, choices, costs, effects, readmission_cost)
    summary = summarize_assignments(assignments, budget)
    summary.update(summary_solver)
    summary["interventions_by_tier"] = assignments.loc[assignments.selected, "intervention"].value_counts().to_dict()
    return {"assignments": assignments, "summary": summary}


def patient_allocation_cohort(test_predictions):
    """Spend once per held-out patient, retaining their latest recorded encounter."""
    required = {"patient_nbr", "encounter_id", "probability"}
    missing = required - set(test_predictions.columns)
    if missing:
        raise ValueError(f"Missing allocation input columns: {sorted(missing)}")
    return (test_predictions.sort_values("encounter_id", kind="stable")
            .drop_duplicates("patient_nbr", keep="last").reset_index(drop=True))


def compare_allocations(cohort, budget, costs=(50, 250, 600),
                        effectiveness=(.15, .25, .35), readmission_cost=15000,
                        objective="prevented", time_limit=15.0):
    """Equal-budget cheap-tier comparators plus an explicitly infeasible blanket."""
    p, costs, effects = _validate(cohort.probability, budget, costs, effectiveness, readmission_cost)
    optimized = allocate_budget(p, budget, costs, effects, readmission_cost, objective, time_limit)
    records = [{"strategy": "Optimized multiple-choice", **optimized["summary"]}]
    cheap = int(np.argmin(costs))
    covered = min(len(p), int((budget + 1e-8) // costs[cheap]))
    rng = np.random.default_rng(SEED)
    orders = {
        "Random (seed 42)": rng.permutation(len(p))[:covered],
        "First-come (encounter ID proxy)": np.argsort(cohort.encounter_id.to_numpy(), kind="stable")[:covered],
        "Top risk (cheapest tier)": np.argsort(-p, kind="stable")[:covered],
        "Blanket (all patients; may exceed budget)": np.arange(len(p)),
    }
    for name, selected in orders.items():
        choices = np.full(len(p), -1, dtype=int)
        choices[selected] = cheap
        assignment = _assignments(p, choices, costs, effects, readmission_cost)
        records.append({"strategy": name, **summarize_assignments(assignment, budget),
                        "solution_status": "baseline", "objective": objective})
    lottery_prevented = float(p.sum() * (covered / len(p) if len(p) else 0) * effects[cheap])
    lottery_spend = float(covered * costs[cheap])
    records.append({
        "strategy": "Equal-budget lottery (expectation)", "patients_selected": covered,
        "spending": lottery_spend, "budget": float(budget), "budget_feasible": True,
        "expected_prevented_readmissions": lottery_prevented,
        "expected_avoided_cost": lottery_prevented * readmission_cost,
        "expected_net_savings": lottery_prevented * readmission_cost - lottery_spend,
        "roi": (lottery_prevented * readmission_cost - lottery_spend) / lottery_spend if lottery_spend else None,
        "solution_status": "analytical random-allocation expectation", "objective": objective,
    })
    return optimized, pd.DataFrame(records)


def run_allocation(test_predictions, output_dir="."):
    """Export the reproducible default scenario and a budget sensitivity curve."""
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    for folder in ("artifacts", "tables", "figures"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    cohort = patient_allocation_cohort(test_predictions)
    budget = float(np.floor(.10 * len(cohort)) * 50)
    optimized, comparison = compare_allocations(cohort, budget)
    assigned = pd.concat([cohort.drop(columns="probability"), optimized["assignments"]], axis=1)
    assigned.to_csv(output / "tables/allocation_assignments.csv", index=False)
    comparison.drop(columns=["interventions_by_tier"], errors="ignore").to_csv(output / "tables/budget_comparison.csv", index=False)
    sensitivity = []
    for fraction in (.02, .05, .10, .20, .40, .60, 1.0):
        scenario_budget = float(np.floor(fraction * len(cohort)) * 50)
        solution = optimized if scenario_budget == budget else allocate_budget(cohort.probability, scenario_budget, time_limit=5)
        sensitivity.append({"cheap_tier_coverage_fraction": fraction, **solution["summary"]})
    sensitivity_df = pd.DataFrame(sensitivity).drop(columns="interventions_by_tier", errors="ignore")
    sensitivity_df.to_csv(output / "tables/budget_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(sensitivity_df.budget, sensitivity_df.expected_prevented_readmissions, "o-", color="#0b9f96")
    ax.set(xlabel="Follow-up budget ($)", ylabel="Feasible expected prevented readmissions", title="Budget sensitivity under assumed intervention effects")
    ax.grid(alpha=.2)
    fig.text(.05, .01, "Solver status is saved per scenario; time limits may lead to feasible heuristic solutions.", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(output / "figures/budget_sensitivity.png", dpi=180)
    plt.close(fig)
    metrics = {
        "scenario_only": True,
        "interpretation": "Expected benefits under hypothetical relative risk reductions; these are not observed intervention outcomes or causal estimates.",
        "allocation_unit": "One latest held-out encounter per patient, ordered by encounter_id (time proxy).",
        "test_encounters": int(len(test_predictions)), "unique_test_patients": int(len(cohort)),
        "duplicate_patient_encounters_removed": int(len(test_predictions) - len(cohort)),
        "assumptions": {"tier_names": list(TIER_NAMES), "costs": [50, 250, 600],
                        "effectiveness": [.15, .25, .35], "readmission_cost": 15000,
                        "budget": budget, "default_coverage": "10% of patients at the cheapest tier"},
        "optimized": optimized["summary"],
        "comparison": comparison.drop(columns="interventions_by_tier", errors="ignore").replace({np.nan: None}).to_dict("records"),
        "sensitivity": sensitivity_df.replace({np.nan: None}).to_dict("records"),
    }
    with open(output / "artifacts/budget_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    return metrics
