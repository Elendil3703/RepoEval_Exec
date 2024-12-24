import unittest
import json
import os
import sys
import re
import numpy as np  # Ensure numpy is available in the injected environment
from scipy import interpolate  # Ensure scipy is available for interpolation
import jax.numpy as jnp  # Ensure jax.numpy is imported for consistency

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInterpolateOutliersFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[342]  # Get the 343rd JSON element (0-indexed -> 342)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the selected JSON array")

    def test_interpolate_outliers(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Gather results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Static checks -------------------
                # 1) Static Check: Ensure the function 'interpolate_outliers' is defined
                if "def interpolate_outliers" not in code:
                    print(f"Code snippet {i}: FAILED, 'interpolate_outliers' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "interpolate_outliers",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+interpolate_outliers\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'interpolate_outliers'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "interpolate_outliers",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution and logic testing -------------------
                exec_globals = {
                    'jnp': jnp,
                    'interpolate': interpolate,
                    'np': np  # Inject necessary modules
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure the function 'interpolate_outliers' exists after execution
                    if 'interpolate_outliers' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'interpolate_outliers' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "interpolate_outliers",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Get the 'interpolate_outliers' function
                    interpolate_outliers = exec_locals['interpolate_outliers']

                    # Testing examples
                    x = jnp.array([1, 2, 100, 4, 5])
                    outlier_idx = jnp.array([2])
                    expected_x = np.array([1, 2, 3, 4, 5])

                    # Run the function and validate output
                    result_x = np.array(interpolate_outliers(x, outlier_idx))
                    np.testing.assert_array_almost_equal(
                        result_x,
                        expected_x,
                        err_msg=f"Code snippet {i} did not interpolate outliers as expected."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "interpolate_outliers",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "interpolate_outliers",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write results to test_result.jsonl =============
        # Load existing records from test_result.jsonl (if exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "interpolate_outliers"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "interpolate_outliers"
        ]

        # Append new results
        existing_records.extend(results)

        # Re-write test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()