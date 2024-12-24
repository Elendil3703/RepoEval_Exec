import unittest
import json
import os
import sys
import numpy as np
import jax.numpy as jnp
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPmfFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        # Get the 341st code snippet (0-indexed, so index 340)
        cls.code_snippet = data[340]
        if not cls.code_snippet:
            raise ValueError("Expected a valid code snippet in the JSON data")

    def test_pmf_function(self):
        """Test the _pmf function with various inputs."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []  # Collect results to write to JSONL

        code = self.code_snippet
        exec_globals = {
            'jnp': jnp,
            'np': np,
            'sys': sys,
            'Any': Any,
        }
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)
            _pmf = exec_locals.get('_pmf')

            if not _pmf:
                raise ValueError("'_pmf' function is not defined in the provided code snippet.")

            # Define test cases
            test_cases = [
                {
                    'p': jnp.array([0.1, 0.2, 0.3]),
                    'x': jnp.array([0.1, 0.2, 0.3]),
                    'expected': np.array([1/3, 1/3, 1/3]),
                },
                {
                    'p': jnp.array([0.1, 0.2, 0.5, 0.5, 0.7]),
                    'x': jnp.array([0.1, 0.2, 0.5, 0.7]),
                    'expected': np.array([1/5, 1/5, 2/5, 1/5]),
                },
                {
                    'p': jnp.array([]),
                    'x': jnp.array([0.1, 0.2, 0.3]),
                    'expected': np.array([0, 0, 0]),
                }
            ]

            for i, test_case in enumerate(test_cases):
                p, x, expected = test_case['p'], test_case['x'], test_case['expected']
                with self.subTest(test_case_index=i):
                    print(f"Running test case {i}...")
                    # Run the function
                    result = _pmf(p, x)

                    try:
                        np.testing.assert_allclose(result, expected, rtol=1e-5)
                        print(f"Test case {i}: PASSED.")
                        passed_count += 1
                        results.append({
                            "function_name": "_pmf",
                            "code": code,
                            "result": "passed",
                            "test_case_index": i,
                        })
                    except AssertionError as e:
                        print(f"Test case {i}: FAILED with error: {e}")
                        failed_count += 1
                        results.append({
                            "function_name": "_pmf",
                            "code": code,
                            "result": "failed",
                            "test_case_index": i,
                        })

        except Exception as e:
            print(f"Execution error: {e}")
            failed_count += len(test_cases)
            results.append({
                "function_name": "_pmf",
                "code": code,
                "result": "failed",
                "error": str(e),
            })

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {passed_count + failed_count}")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records concerning "_pmf"
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "_pmf"]

        # Append new results
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()