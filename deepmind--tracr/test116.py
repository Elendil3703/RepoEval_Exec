import unittest
import json
import sys
import os
from typing import Any  

TEST_RESULT_JSONL = "test_result.jsonl"

class TestAnnotateFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[115]  # Get the 116th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Tests the 'annotate' function in the provided code snippets."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Testing code snippet {i}...")

                exec_globals = {
                    'sys': sys,
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'annotate' function exists
                    if 'annotate' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'annotate' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "annotate",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Dummy RASPExprT class for testing
                    class RASPExprT:
                        def __init__(self):
                            self.annotations = {}

                        def copy(self):
                            return RASPExprT()

                    expr = RASPExprT()
                    expr.annotations = {"key1": "value1"}

                    # Use the 'annotate' function with additional annotations
                    annotated_expr = exec_locals['annotate'](expr, key2="value2", key1="new_value")

                    # Tests: ensure annotations are correctly updated
                    self.assertEqual(
                        annotated_expr.annotations.get("key1"), 
                        "new_value", 
                        f"Code snippet {i} failed to update 'key1'."
                    )
                    self.assertEqual(
                        annotated_expr.annotations.get("key2"), 
                        "value2", 
                        f"Code snippet {i} failed to add 'key2'."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "annotate",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "annotate",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
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

        # Remove old records for the function name 'annotate'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "annotate"
        ]

        # Add new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()