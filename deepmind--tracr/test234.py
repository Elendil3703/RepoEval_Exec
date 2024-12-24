import unittest
import json
import os
import re
from typing import Dict

TEST_RESULT_JSONL = "test_result.jsonl"

class TestLayerNaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[233]  # Get the specified JSON element
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected non-empty code snippet in JSON data")

    def test_check_layer_naming(self):
        """Dynamically test the provided code snippet for _check_layer_naming."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results for writing to JSONL

        code = self.code_snippet

        with self.subTest():
            print("Running test for _check_layer_naming...")

            # Static check for _check_layer_naming definition
            if "def _check_layer_naming" not in code:
                print("Code snippet: FAILED, '_check_layer_naming' function not found.\n")
                failed_count += 1
                # Write failure record
                results.append({
                    "function_name": "_check_layer_naming",
                    "code": code,
                    "result": "failed"
                })
            else:
                # Prepare for dynamic execution
                exec_globals = {
                    'self': self
                }
                exec_locals = {}

                try:
                    # Prepare a mock implementation of assertions
                    class MockTest(unittest.TestCase):
                        def assertStartsWith(self, value, prefix):
                            if not value.startswith(prefix):
                                raise AssertionError(f"{value} does not start with {prefix}")

                    mock_self = MockTest()

                    # Dynamically execute and test code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check _check_layer_naming is present
                    if '_check_layer_naming' not in exec_locals:
                        print("Code snippet: FAILED, '_check_layer_naming' not in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_check_layer_naming",
                            "code": code,
                            "result": "failed"
                        })
                    else:
                        # Invoke _check_layer_naming on test data
                        test_params = {
                            "compressed_transformer/layer_1/mlp/linear_1": {},
                            "compressed_transformer/layer_1/attn/key": {},
                            "compressed_transformer/layer_norm": {},
                        }
                        exec_locals['_check_layer_naming'](mock_self, test_params)

                        print("Code snippet: PASSED all assertions.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "_check_layer_naming",
                            "code": code,
                            "result": "passed"
                        })
                except Exception as e:
                    print(f"Code snippet: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_check_layer_naming",
                        "code": code,
                        "result": "failed"
                    })

        # Test Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for _check_layer_naming
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_check_layer_naming"
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