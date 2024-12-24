import unittest
import json
import sys
import os
import re
import jax.numpy as jnp
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestIntraWeekSeasonality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and extract the 310th element's code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[309]  # Get the 310th element

        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the 310th JSON array")

    def test_intra_week_seasonality(self):
        """Dynamically test the intra_week_seasonality function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL
        
        code = self.code_snippet
        
        def mock_priors():
            # Mock priors or methods needed from priors for testing
            class MockPriors:
                WEEKDAY = "weekday"
                
                @staticmethod
                def get_default_priors():
                    return {MockPriors.WEEKDAY: lambda: jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])}
            return MockPriors()

        def mock_numpyro_sample(name, fn):
            return fn()

        exec_globals = {
            'sys': sys,
            'numpyro': type('numpyro', (), {'sample': mock_numpyro_sample, 'plate': lambda name, size: range(size)}),
            'priors': mock_priors(),
            'jnp': jnp,
            'custom_priors': {'weekday': lambda: jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])},
            '_intra_week_seasonality': lambda data, weekday: data * weekday
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Verify function and logic
            if 'intra_week_seasonality' not in exec_locals:
                raise AssertionError("Function 'intra_week_seasonality' not found.")

            # Prepare test data
            data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            
            # Execute function
            weekday_series = exec_locals['intra_week_seasonality'](data=data)
            
            # Expected outcome based on mock setup
            expected_weekday_series = data * jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            if data.ndim == 3:
                expected_weekday_series = jnp.expand_dims(expected_weekday_series, axis=-1)
            
            # Assertions
            jnp.testing.assert_allclose(weekday_series, expected_weekday_series,
                                        err_msg="Weekday series does not match expected values.")

            print("intra_week_seasonality: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "intra_week_seasonality",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"intra_week_seasonality: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "intra_week_seasonality",
                "code": code,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with the same function name
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "intra_week_seasonality"
        ]

        # Append new results
        existing_records.extend(results)

        # Write updated records
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()