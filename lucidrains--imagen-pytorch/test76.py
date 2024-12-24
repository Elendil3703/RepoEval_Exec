import unittest
import json
import os
import sys
from typing import Any  # Ensure the injected environment has Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file containing the code snippets
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[75]  # Get the 76th JSON element (index 75)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 76th JSON array")

    def test_init_function(self):
        """Dynamically test all code snippets in the JSON related to the __init__ function."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static checks for the __init__ function presence
                if "def __init__" not in code:
                    print(f"Code snippet {i}: FAILED, '__init__' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'nn': __import__('torch').nn,
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Execute the code snippet dynamically
                    exec(code, exec_globals, exec_locals)

                    # Check if the class that contains __init__ is defined
                    class_name = [name for name in exec_locals if isinstance(exec_locals[name], type)]
                    if not class_name:
                        print(f"Code snippet {i}: FAILED, no class with __init__ found.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    class_instance = exec_locals[class_name[0]](dim=64, dim_out=128)

                    # Test if attributes are properly set
                    self.assertTrue(
                        hasattr(class_instance, 'time_mlp'),
                        f"Code snippet {i}: 'time_mlp' attribute not found."
                    )
                    self.assertTrue(
                        hasattr(class_instance, 'cross_attn'),
                        f"Code snippet {i}: 'cross_attn' attribute not found."
                    )
                    
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

        # Print a summary of the tests
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records related to "__init__"
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