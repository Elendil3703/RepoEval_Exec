import unittest
import json
import sys
import re
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestOptimizerTypeFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[287]  # Get the 288th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 288th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON for get_optimizer_type function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect the test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Basic static checks
                if "get_optimzer_type" not in code:
                    print(f"Code snippet {i}: FAILED, 'get_optimzer_type' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_optimzer_type",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+get_optimzer_type\s*\(optimizer\)"
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'get_optimzer_type'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_optimzer_type",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and test logic
                exec_globals = {
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if the function is correctly defined
                    if 'get_optimzer_type' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'get_optimzer_type' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "get_optimzer_type",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Function to test
                    get_optimzer_type = exec_locals['get_optimzer_type']

                    # Define a mock optimizer class for testing
                    class MockAdamOptimizer:
                        # Simulate optimizer with 'adam' in class name
                        pass

                    class MockSGDOptimizer:
                        # Simulate optimizer without 'adam' in class name
                        pass

                    # Test cases
                    adam_result = get_optimzer_type(MockAdamOptimizer())  # Should return 'adam'
                    sgd_result = get_optimzer_type(MockSGDOptimizer())    # Should return 'sgd'
                    
                    # Perform test assertions
                    self.assertEqual(adam_result, "adam", f"Code snippet {i}: FAILED in getting 'adam'.\n")
                    self.assertEqual(sgd_result, "sgd", f"Code snippet {i}: FAILED in getting 'sgd'.\n")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "get_optimzer_type",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_optimzer_type",
                        "code": code,
                        "result": "failed"
                    })

        # Print test summary
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

        # Remove old records for 'get_optimzer_type'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "get_optimzer_type"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()