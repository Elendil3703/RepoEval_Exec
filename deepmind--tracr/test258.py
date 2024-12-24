import unittest
import json
import sys
import os
import warnings
from typing import Any

# Constants
TEST_RESULT_JSONL = "test_result.jsonl"
DATA_INDEX = 257

class TestInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[DATA_INDEX]  # Get the targeted JSON element
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected non-empty code snippet at index 257")

    def test_init_function(self):
        """Test the __init__ method with multiple scenarios."""
        passed_count = 0
        failed_count = 0
        results = []

        # Sample test parameters
        test_params = [
            {"num_heads": 8, "key_size": 64, "value_size": 128, "model_size": 512, "w_init_scale": None, "w_init": object()},
            {"num_heads": 4, "key_size": 32, "value_size": None, "model_size": None, "w_init_scale": 0.5, "w_init": None},
            {"num_heads": 2, "key_size": 16, "value_size": 16, "model_size": 64, "w_init_scale": None, "w_init": None},
        ]

        for i, params in enumerate(test_params):
            with self.subTest(params_index=i):
                print(f"Running test case {i}...")
                exec_globals = {"Any": Any, "warnings": warnings}
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(self.code_snippet, exec_globals, exec_locals)

                    if '__init__' not in exec_locals:
                        print(f"Error: '__init__' not defined in the code snippet for test case {i}.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code_index": DATA_INDEX,
                            "result": "failed"
                        })
                        continue

                    init_function = exec_locals['__init__']

                    # Create instance and check attributes
                    obj = type("TestClass", (object,), {"__init__": init_function})
                    instance = obj(**params)

                    # Validate attributes
                    self.assertEqual(instance.num_heads, params['num_heads'])
                    self.assertEqual(instance.key_size, params['key_size'])
                    self.assertEqual(instance.value_size, params.get('value_size') or params['key_size'])
                    
                    expected_model_size = params.get('model_size') or (params['key_size'] * params['num_heads'])
                    self.assertEqual(instance.model_size, expected_model_size)

                    if params['w_init'] and params['w_init_scale']:
                        self.assertRaises(ValueError)

                    if not params['w_init'] and not params['w_init_scale']:
                        self.assertRaises(ValueError)

                    print(f"Test case {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code_index": DATA_INDEX,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Test case {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code_index": DATA_INDEX,
                        "result": "failed"
                    })

        # Report test results
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_params)}\n")
        self.assertEqual(passed_count + failed_count, len(test_params), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove existing records for the same function
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "__init__"]

        # Append new results
        existing_records.extend(results)

        # Write updated records
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()