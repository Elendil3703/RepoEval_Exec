import unittest
import json
import os
import torch
from torch import nn

TEST_RESULT_JSONL = "test_result.jsonl"

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

def exists(val):
    return val is not None

class TestInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[67]  # Get the 68th JSON element, index 67
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in this JSON array")

    def test_initialization(self):
        """Dynamically test the initialization code snippet from the JSON with various configurations."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results for writing to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                try:
                    print(f"Running test for code snippet {i}...")

                    # Execute the provided class
                    exec_globals = {
                        '__name__': '__main__',
                        'torch': torch,
                        'nn': nn,
                        'LayerNorm': LayerNorm,
                        'exists': exists
                    }
                    exec_locals = {}

                    exec(code, exec_globals, exec_locals)

                    # Assume the class name is 'TestModule'
                    if 'TestModule' not in exec_locals:
                        raise ValueError(f"Code snippet {i}: 'TestModule' class not found in the execution environment.")

                    TestModuleClass = exec_locals['TestModule']

                    # Test with different parameters
                    instance = TestModuleClass(dim=64)
                    self.assertEqual(instance.heads, 8)
                    self.assertEqual(instance.scale, 8)
                    self.assertIsNotNone(instance.norm)
                    self.assertIsInstance(instance.to_q, nn.Linear)

                    if instance.to_context is not None:
                        self.assertIsInstance(instance.to_context, nn.Sequential)
                        
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })
                except AssertionError as e:
                    print(f"Code snippet {i}: FAILED with assertion error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
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

        # Remove old records for this function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # Append new results
        existing_records.extend(results)

        # Write to JSONL
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()