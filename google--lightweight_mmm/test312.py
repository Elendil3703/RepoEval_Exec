import unittest
import json
import sys
import os
import jax.numpy as jnp
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestDynamicTrend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[311]  # Get the 312th element (0-indexed)

    def test_dynamic_trend(self):
        """Test the _dynamic_trend function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Dynamic execution and testing -------------------
                exec_globals = {
                    'jnp': jnp,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if _dynamic_trend is defined in exec_locals
                    if '_dynamic_trend' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_dynamic_trend' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_dynamic_trend",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Generate test inputs
                    number_periods = 5
                    random_walk_level = jnp.array([0.1, 0.2, -0.1, 0.0, 0.3])
                    random_walk_slope = jnp.array([0.05, -0.05, 0.1, 0.0, -0.1])
                    initial_level = jnp.array([1.0])
                    initial_slope = jnp.array([0.5])
                    variance_level = jnp.array([2.0])
                    variance_slope = jnp.array([1.0])

                    # Call the _dynamic_trend function
                    result = exec_locals['_dynamic_trend'](
                        number_periods, random_walk_level,
                        random_walk_slope, initial_level,
                        initial_slope, variance_level, variance_slope
                    )

                    # Run assertions
                    expected_output_shape = (5,)
                    self.assertEqual(
                        result.shape,
                        expected_output_shape,
                        f"Code snippet {i}: Output shape mismatch, expected {expected_output_shape}, got {result.shape}."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_dynamic_trend",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_dynamic_trend",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for _dynamic_trend
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_dynamic_trend"
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