import unittest
import json
import sys
import os
from typing import Any, Tuple
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPlotCrossCorrelate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[298]  # Get the 299th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the specified JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "def plot_cross_correlate" not in code:
                    print(f"Code snippet {i}: FAILED, 'plot_cross_correlate' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "plot_cross_correlate",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+plot_cross_correlate\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'plot_cross_correlate'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "plot_cross_correlate",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'jnp': jnp,
                    'plt': plt,
                    'Tuple': Tuple
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    if 'plot_cross_correlate' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'plot_cross_correlate' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "plot_cross_correlate",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    plot_cross_correlate = exec_locals['plot_cross_correlate']

                    feature = jnp.array([1, 2, 3, 4, 5])
                    target = jnp.array([5, 4, 3, 2, 1])
                    maxlags = 2

                    try:
                        lag_index, correlation = plot_cross_correlate(feature, target, maxlags)
                        self.assertIsInstance(lag_index, int, "Lag index should be of type int.")
                        self.assertIsInstance(correlation, float, "Correlation should be of type float.")

                        print(f"Code snippet {i}: PASSED all assertions.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "plot_cross_correlate",
                            "code": code,
                            "result": "passed"
                        })
                    except ValueError as e:
                        print(f"Code snippet {i}: FAILED with ValueError: {e}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "plot_cross_correlate",
                            "code": code,
                            "result": "failed"
                        })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "plot_cross_correlate",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "plot_cross_correlate"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()