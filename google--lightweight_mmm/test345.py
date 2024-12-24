import unittest
import json
import sys
import os
from typing import Any  # Ensure Any is available in the execution environment
import numpyro
import models  # Assuming models is a module containing transform_hill_adstock
import numpy as np

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMockModelFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[344]  # Get the 345th JSON element (index 344)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 345th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static checks: Ensure function existence and signature
                if "mock_model_function" not in code:
                    print(f"Code snippet {i}: FAILED, 'mock_model_function' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Check for the deterministic and transformation logic presence
                if "numpyro.deterministic" not in code or "models.transform_hill_adstock" not in code:
                    print(f"Code snippet {i}: FAILED, 'numpyro.deterministic' or 'models.transform_hill_adstock' not correctly used.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'numpyro': numpyro,
                    'models': models,
                    'np': np,
                }
                exec_locals = {}

                try:
                    # Dynamic execution of the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check existence of mock_model_function
                    if 'mock_model_function' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'mock_model_function' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "mock_model_function",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the mock_model_function with simulated data
                    sample_data = np.array([1.0, 2.0, 3.0])
                    exec_locals['mock_model_function'](sample_data)

                    # Assertion or checks can be more specific based on expected behavior
                    # For example, you could mock transform_hill_adstock to return a known value

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name == "mock_model_function"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "mock_model_function"
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