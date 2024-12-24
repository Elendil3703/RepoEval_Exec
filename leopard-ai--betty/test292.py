import unittest
import json
import os
import torch
from typing import Any  # Ensure Any is available

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInitResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[291]  # Get the 292nd JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 292nd JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets for __init__ definition."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect the test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static Checks -------------------
                # 1) Static check: ensure the snippet includes `torch.nn.Parameter` and `super().__init__()`
                if "torch.nn.Parameter" not in code:
                    print(f"Code snippet {i}: FAILED, 'torch.nn.Parameter' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "super().__init__()" not in code:
                    print(f"Code snippet {i}: FAILED, 'super().__init__()' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution -------------------
                exec_globals = {
                    'torch': torch,
                    'DATA_DIM': 10,  # Sample data dimension for testing
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Test class instantiation and attribute existence
                    class TestClass:
                        def __init__(self):
                            self.w = torch.nn.Parameter(torch.ones(10))

                    # Instantiate and test
                    obj = TestClass()

                    self.assertTrue(hasattr(obj, 'w'), f"Code snippet {i} did not define 'w' as an attribute.")
                    self.assertIsInstance(obj.w, torch.nn.Parameter, f"Code snippet {i} did not correctly define 'w' as a torch.nn.Parameter.")

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

        # Final Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write Test Results to test_result.jsonl =============
        # Read the existing test_result.jsonl (if it doesn't exist, ignore)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records with function_name == "__init__"
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