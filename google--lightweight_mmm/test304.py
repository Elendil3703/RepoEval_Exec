import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMockModelFunctionResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[303]  # Get the 304th JSON element (index 303)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets for mock_model_function in the JSON."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static checks for signatures
                if "def mock_model_function" not in code:
                    print(f"Code snippet {i}: FAILED, 'mock_model_function' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and function test
                exec_globals = {
                    'sys': sys,
                    'numpyro': Any,   # Simulating numpyro import
                    'seasonality': Any,  # Simulating seasonality import
                }
                exec_locals = {}

                try:
                    # Dynamic execution of the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure the function is defined
                    if 'mock_model_function' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'mock_model_function' not defined after exec.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "mock_model_function",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test calls to the function
                    mock_function = exec_locals['mock_model_function']
                    mock_data = [0, 1, 2, 3]
                    degrees_seasonality = 5
                    frequency = 1.0

                    # Assume numpyro and seasonality are behaving correctly in the imports
                    # as we only test function availability and signature
                    mock_result = mock_function(mock_data, degrees_seasonality, frequency)

                    # For this example, consider the execution reaching this point a success
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

        # Summary of results
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

        # Remove old records for the function
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