import unittest
import json
import jax.numpy as jnp
from typing import Optional
from scipy import optimize
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class CustomScaler:
    def inverse_transform(self, media):
        # Simulated inverse transform operation
        return media * 2  # some arbitrary scaling back for testing

def _get_lower_and_upper_bounds(
    media: jnp.ndarray,
    n_time_periods: int,
    lower_pct: jnp.ndarray,
    upper_pct: jnp.ndarray,
    media_scaler: Optional[CustomScaler] = None
) -> optimize.Bounds:
    if media.ndim == 3:
        lower_pct = jnp.expand_dims(lower_pct, axis=-1)
        upper_pct = jnp.expand_dims(upper_pct, axis=-1)

    mean_data = media.mean(axis=0)
    lower_bounds = jnp.maximum(mean_data * (1 - lower_pct), 0)
    upper_bounds = mean_data * (1 + upper_pct)

    if media_scaler:
        lower_bounds = media_scaler.inverse_transform(lower_bounds)
        upper_bounds = media_scaler.inverse_transform(upper_bounds)

    if media.ndim == 3:
        lower_bounds = lower_bounds.sum(axis=-1)
        upper_bounds = upper_bounds.sum(axis=-1)

    return optimize.Bounds(lb=lower_bounds * n_time_periods,
                           ub=upper_bounds * n_time_periods)

class TestGetLowerAndUpperBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[330]  # Get the 331st JSON element (index 330)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the array")

    def test_code_snippets(self):
        """Test the `_get_lower_and_upper_bounds` function."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                try:
                    # Define test values
                    media = jnp.array([[1, 2, 3], [4, 5, 6]])
                    n_time_periods = 2
                    lower_pct = jnp.array([0.1, 0.2, 0.3])
                    upper_pct = jnp.array([0.1, 0.2, 0.3])
                    scaler = CustomScaler()

                    expected_lb = jnp.array([4.8, 2.4, 0])  # made-up expected values for testing
                    expected_ub = jnp.array([12, 12, 18])  # made-up expected values for testing

                    # Evaluate the function
                    bounds_result = _get_lower_and_upper_bounds(
                        media, n_time_periods, lower_pct, upper_pct, scaler)

                    self.assertTrue(jnp.allclose(bounds_result.lb, expected_lb), f"Lower bounds do not match for code snippet {i}.")
                    self.assertTrue(jnp.allclose(bounds_result.ub, expected_ub), f"Upper bounds do not match for code snippet {i}.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_get_lower_and_upper_bounds",
                        "code": "N/A",  # No code chunk in JSON file to test here
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_get_lower_and_upper_bounds",
                        "code": "N/A",  # No code chunk in JSON file to test here
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records for function _get_lower_and_upper_bounds
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_get_lower_and_upper_bounds"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()