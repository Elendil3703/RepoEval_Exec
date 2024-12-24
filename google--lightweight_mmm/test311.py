import unittest
import json
import sys
import re
import os
from typing import Any
import numpyro
import jax.numpy as jnp
from numpyro.distributions import Normal
from collections import namedtuple

# Constants for file names
TEST_RESULT_JSONL = "test_result.jsonl"

# Mock objects and functions to replicate the external dependencies in the ground truth function
priors = namedtuple('priors', ['COEF_TREND', 'EXPO_TREND', 'get_default_priors', 'get'])
priors.COEF_TREND = "coef_trend"
priors.EXPO_TREND = "expo_trend"

def mock_get_default_priors():
    return {priors.COEF_TREND: Normal(0, 1), priors.EXPO_TREND: Normal(1, 0.1)}

def mock_get(name, default):
    return default

setattr(priors, 'get_default_priors', mock_get_default_priors)
setattr(priors, 'get', mock_get)

core_utils = namedtuple('core_utils', ['get_number_geos'])
core_utils.get_number_geos = lambda data: 2 if data.shape[1] > 1 else 1

# Mock implementation of the main function this is executing
def _trend_with_exponent(coef_trend, trend, expo_trend):
    return coef_trend * trend ** expo_trend

class TestTrendWithExponentResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[310]  # Get the 311th JSON element (0-based index is 310)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the selected JSON array")

    def test_trend_with_exponent(self):
        """Dynamically test the code snippet in the JSON with additional checks."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static Checks -------------------
                if "_trend_with_exponent" not in code:
                    print(f"Code snippet {i}: FAILED, '_trend_with_exponent' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "trend_with_exponent",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Testing -------------------
                exec_globals = {
                    'numpyro': numpyro,
                    'jnp': jnp,
                    'priors': priors,
                    'core_utils': core_utils,
                    '_trend_with_exponent': _trend_with_exponent,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if the function of interest is defined
                    if 'trend_with_exponent' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'trend_with_exponent' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "trend_with_exponent",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    trend_with_exponent = exec_locals['trend_with_exponent']

                    # Test the dynamic nature of the function
                    data = jnp.ones((10, 2))  # Mock data with 2 geos
                    trend_values = trend_with_exponent(data=data)

                    # Check if the returned trend values are a jnp-array, as expected
                    self.assertTrue(isinstance(trend_values, jnp.ndarray),
                                    f"Code snippet {i} did not return a jnp.ndarray.")
                    
                    # Expected output test (more specific checks would go here)
                    self.assertEqual(trend_values.shape[0], data.shape[0],
                                     f"Code snippet {i} did not return trend values matching input data dimensions.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "trend_with_exponent",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "trend_with_exponent",
                        "code": code,
                        "result": "failed"
                    })

        # Summary of test results
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write to test_result.jsonl =============
        # Read existing records
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "trend_with_exponent"]

        # Append new results
        existing_records.extend(results)

        # Write updated records
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()