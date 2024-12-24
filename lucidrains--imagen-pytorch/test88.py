import unittest
import json
import sys
import os
from typing import Any
import torch
from torch import nn

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGetUnetFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[87]  # Get the 88th JSON element (index 87)

    def test_code_snippets(self):
        """Dynamically test all code snippets for the get_unet function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Dynamic execution and testing logic
                exec_globals = {
                    '__name__': '__main__',
                    'nn': nn,
                    'torch': torch,
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'get_unet' is defined
                    if 'get_unet' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'get_unet' not found.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "get_unet",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Set up a mock object with required attributes for testing get_unet
                    class MockModel:
                        def __init__(self, num_unets, device='cpu'):
                            self.device = device
                            self.unet_being_trained_index = -1
                            self.unets = nn.ModuleList([nn.Linear(10, 10) for _ in range(num_unets)])

                        get_unet = exec_locals['get_unet']

                    # Create a mock model with 3 unets
                    mock_model = MockModel(3)

                    # Test retrieving each unet
                    for unet_number in range(1, 4):
                        unet = mock_model.get_unet(unet_number)
                        self.assertEqual(unet.to(mock_model.device), unet)
                        self.assertEqual(mock_model.unet_being_trained_index, unet_number - 1, "Failed index update")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "get_unet",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_unet",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for "get_unet"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "get_unet"
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