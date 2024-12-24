import unittest
import json
import sys
import os
import torch
import torch.distributed as dist
from typing import Tuple

TEST_RESULT_JSONL = "test_result.jsonl"

class TestUpdateFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[20]  # Get the 21st JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 21st JSON array")

    def test_update_function(self):
        """Test the `update` function from code snippets."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        # Mock class with update and required attributes
        class RunningStats:
            def __init__(self):
                self.mean = torch.tensor(0.0)
                self.var = torch.tensor(0.0)
                self.std = torch.tensor(0.0)
                self.count = 0

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                exec_globals = {
                    'torch': torch,
                    'dist': dist,
                    'get_global_statistics': lambda xs: (xs.mean(), xs.var(), xs.numel()),
                    'RunningStats': RunningStats
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    if 'update' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'update' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "update",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the update function
                    stats = RunningStats()
                    update_func = exec_locals['update'].__get__(stats)
                    
                    # Create test input
                    torch.manual_seed(0)
                    xs = torch.randn(100)
                    
                    expected_mean, expected_var = xs.mean(), xs.var(unbiased=False)

                    # Call update and check results
                    actual_mean, actual_std = update_func(xs)

                    self.assertTrue(torch.isclose(stats.mean, expected_mean, atol=1e-5), f"Code snippet {i}: mean mismatch")
                    self.assertTrue(torch.isclose(actual_mean, expected_mean, atol=1e-5), f"Code snippet {i}: return mean mismatch")
                    self.assertTrue(torch.isclose(actual_std, (expected_var * xs.numel() / (xs.numel() - 1)).sqrt(), atol=1e-5), f"Code snippet {i}: std mismatch")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "update",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "update",
                        "code": code,
                        "result": "failed"
                    })

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

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "update"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()