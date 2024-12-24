import unittest
import json
import os
from typing import Dict

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[393]  # Get the 394th JSON element

    def test_call_function(self):
        """Test the ground truth function __call__ with various input cases."""
        passed_count = 0
        failed_count = 0
        results = []

        # Here, we define our test cases for `__call__`
        test_cases = [
            (
                {"key1": 1, "key2": 2, "key3": 3},
                {"key1": 1, "key3": 3},
                ['key2']
            ),
            (
                {"a": 10, "b": 20, "c": 30},
                {"a": 10, "c": 30},
                ['b']
            ),
            (
                {"x": 5, "y": 10},
                {"y": 10},
                ['x']
            )
        ]

        # Injected helper function
        def _unix_pattern_to_parameter_names(pattern, keys):
            return set([k for k in keys if k in pattern])

        for i, (input_dict, expected_output, exclude_pattern) in enumerate(test_cases):
            with self.subTest(test_index=i):
                print(f"Running test case {i}...")

                exec_globals = {
                    '_unix_pattern_to_parameter_names': _unix_pattern_to_parameter_names,
                    'Dict': Dict
                }
                exec_locals = {}
                
                try:
                    # Execute the code snippet
                    exec(self.code_snippet, exec_globals, exec_locals)
                    
                    # Prepare an instance with the required attribute 'key_pattern'
                    class MockClass:
                        def __init__(self, key_pattern):
                            self.key_pattern = key_pattern
                    
                    # Instantiate the class and call the `__call__` method
                    instance = MockClass(exclude_pattern)
                    result = exec_locals['__call__'](instance, input_dict)

                    self.assertEqual(result, expected_output, f"Test case {i} failed: output did not match expected.")

                    passed_count += 1
                    results.append({
                        "function_name": "__call__",
                        "input": input_dict,
                        "expected_output": expected_output,
                        "result": "passed"
                    })

                    print(f"Test case {i}: PASSED.\n")
                except Exception as e:
                    print(f"Test case {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__call__",
                        "input": input_dict,
                        "expected_output": expected_output,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # ============= Write results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for "__call__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__call__"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite jsonl file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()