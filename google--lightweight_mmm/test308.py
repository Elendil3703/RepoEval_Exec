import unittest
import json
import sys
import jax.numpy as jnp
import os

TEST_RESULT_JSONL = "test_result.jsonl"


class TestSinusoidalSeasonality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[307]  # Get the 308th JSON element (zero-indexed)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 308th JSON array")

    def test_sinusoidal_seasonality(self):
        """Test the _sinusoidal_seasonality function for certain conditions."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results for writing to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Inject the necessary globals
                exec_globals = {
                    'jnp': jnp,
                }
                exec_locals = {}

                try:
                    # Execute the code snippet to define the function
                    exec(code, exec_globals, exec_locals)

                    # Check if _sinusoidal_seasonality function is defined
                    if '_sinusoidal_seasonality' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_sinusoidal_seasonality' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_sinusoidal_seasonality",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Define test arrays
                    seasonality_arange = jnp.arange(4)
                    degrees_arange = jnp.arange(2)
                    gamma_seasonality = jnp.array([[0.5, 0.5], [0.5, 0.5]])
                    frequency = 24

                    # Execute the function and check the result
                    result = exec_locals['_sinusoidal_seasonality'](
                        seasonality_arange,
                        degrees_arange,
                        gamma_seasonality,
                        frequency
                    )
                    
                    expected_shape = (4,)
                    self.assertEqual(
                        result.shape, expected_shape, f"Code snippet {i} returned result with unexpected shape."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_sinusoidal_seasonality",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_sinusoidal_seasonality",
                        "code": code,
                        "result": "failed"
                    })

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test result to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for _sinusoidal_seasonality
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_sinusoidal_seasonality"
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