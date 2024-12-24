import unittest
import json
import os
import torch
import torch.nn as nn
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[71]  # Get the 72nd JSON element (index 71)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 72nd JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets for ground truth __init__ method."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Verify snippets contain specific components
                if "super().__init__()" not in code:
                    print(f"Code snippet {i}: FAILED, 'super().__init__()' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "assert (dim % 2) == 0" not in code:
                    print(f"Code snippet {i}: FAILED, assertion on 'dim' is missing.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "self.weights = nn.Parameter(torch.randn(" not in code:
                    print(f"Code snippet {i}: FAILED, 'self.weights' initialization not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Attempt to execute the code dynamically
                exec_globals = {'torch': torch, 'nn': nn, 'Any': Any}
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    class_obj = next(
                        (cls for cls in exec_locals.values() if isinstance(cls, type) and '__init__' in cls.__dict__), 
                        None
                    )

                    # Ensure we found a class definition with __init__ method
                    if class_obj is None:
                        print(f"Code snippet {i}: FAILED, no class with __init__ found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test instantiation with valid even dimension
                    try:
                        instance = class_obj(4)
                        self.assertIsInstance(instance.weights, nn.Parameter)
                        self.assertEqual(instance.weights.shape[0], 2, f"Incorrect parameter shape for code snippet {i}.")
                    except Exception as e:
                        print(f"Code snippet {i}: FAILED during instantiation with dim=4, error: {e}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test instantiation with invalid odd dimension
                    with self.assertRaises(AssertionError):
                        class_obj(3)
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

        # Summary and write results
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

        # Remove old records for __init__
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
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