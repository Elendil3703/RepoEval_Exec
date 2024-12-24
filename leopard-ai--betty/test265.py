import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGetBatchFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[264]  # Get the 265th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 265th JSON array")

    def test_get_batch(self):
        """Dynamically test all code snippets in the JSON for 'get_batch' function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static Check: Ensure 'get_batch' is in the code snippet
                if "def get_batch" not in code:
                    print(f"Code snippet {i}: FAILED, 'get_batch' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_batch",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic Execution and Test Logic
                exec_globals = {
                    'Any': Any,
                    'tuple': tuple
                }
                exec_locals = {}

                try:
                    # Dynamic execution of the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'get_batch' is correctly defined
                    if 'get_batch' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'get_batch' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "get_batch",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create a mock class with necessary methods and attributes
                    class MockDataLoader:
                        def __init__(self, loaders):
                            self.train_data_loader = loaders

                        def get_batch_single_loader(self, i):
                            return f"data_loader_{i}"

                    # Creating an instance with 3 data loaders
                    instance = MockDataLoader(['loader1', 'loader2', 'loader3'])
                    
                    # Executing get_batch from the instance
                    exec_locals['self'] = instance
                    batch = exec_locals['get_batch'](instance)

                    # Assertions to validate the behavior
                    self.assertIsInstance(batch, tuple, f"Code snippet {i} failed. Expected tuple, got {type(batch)}")
                    self.assertEqual(batch, ('data_loader_0', 'data_loader_1', 'data_loader_2'),
                                     f"Code snippet {i} failed. Expected specific loaders in batch.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "get_batch",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_batch",
                        "code": code,
                        "result": "failed"
                    })

        # Summary
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

        # Remove old records for 'get_batch'
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "get_batch"]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()