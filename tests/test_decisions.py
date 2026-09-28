"""Small exact checks for the allocator's clinical-resource constraints."""
import itertools
import unittest

import numpy as np
import pandas as pd

from readmitrisk.allocation import allocate_budget, patient_allocation_cohort


class AllocationTests(unittest.TestCase):
    def test_matches_exhaustive_optimum_for_both_objectives(self):
        probabilities = np.array([0.0, .02, .11, .31, .72])
        costs, effects, avoided_cost = np.array([50, 125, 260]), np.array([.15, .33, .55]), 1000
        for objective in ("prevented", "net_savings"):
            for budget in (0, 50, 175, 385, 800):
                best = 0.0
                for choice in itertools.product((-1, 0, 1, 2), repeat=len(probabilities)):
                    spending = sum(costs[tier] for tier in choice if tier >= 0)
                    if spending > budget:
                        continue
                    benefit = sum(probabilities[i] * effects[tier] for i, tier in enumerate(choice) if tier >= 0)
                    value = benefit if objective == "prevented" else benefit * avoided_cost - spending
                    best = max(best, value)
                result = allocate_budget(probabilities, budget, costs, effects, avoided_cost,
                                         objective=objective, mip_rel_gap=0)
                summary, assignments = result["summary"], result["assignments"]
                measured = summary["expected_prevented_readmissions"] if objective == "prevented" else summary["expected_net_savings"]
                with self.subTest(objective=objective, budget=budget):
                    self.assertAlmostEqual(measured, best, places=7)
                    self.assertLessEqual(summary["spending"], budget)
                    self.assertTrue(assignments.patient_position.is_unique)
                    self.assertTrue(summary["budget_feasible"])

    def test_latest_encounter_keeps_one_allocation_per_patient(self):
        frame = pd.DataFrame({"patient_nbr": [8, 7, 8, 7], "encounter_id": [1, 4, 3, 2],
                              "probability": [.1, .2, .3, .4], "y_true": [0, 1, 0, 1]})
        result = patient_allocation_cohort(frame)
        self.assertEqual(dict(zip(result.patient_nbr, result.encounter_id)), {7: 4, 8: 3})
        self.assertEqual(dict(zip(result.patient_nbr, result.probability)), {7: .2, 8: .3})

    def test_no_spending_for_zero_risk_or_negative_net_value(self):
        zero = allocate_budget([0, 0], 100)
        negative = allocate_budget([.001, .002], 100, readmission_cost=100, objective="net_savings")
        self.assertEqual(zero["summary"]["spending"], 0)
        self.assertEqual(negative["summary"]["spending"], 0)

    def test_invalid_scenario_rejected(self):
        for kwargs in ({"probabilities": [np.nan]}, {"probabilities": [1.1]},
                       {"costs": [0, 250, 600]}, {"effectiveness": [.1, 1.1, .3]}, {"budget": -1}):
            arguments = {"probabilities": [.1, .2], "budget": 100, **kwargs}
            with self.assertRaises(ValueError):
                allocate_budget(**arguments)


if __name__ == "__main__":
    unittest.main()
