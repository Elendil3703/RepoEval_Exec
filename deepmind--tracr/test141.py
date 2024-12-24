import unittest
import json
import sys
import re
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInitMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[140]  # Get the 141st JSON element (140th index)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 141st JSON array")

    def test_init_methods(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # 1) Static check: Ensure __init__ method and _eval_fn_by_expr_type are present
                if "__init__" not in code:
                    print(f"Code snippet {i}: FAILED, '__init__' method not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue
                
                if "_eval_fn_by_expr_type" not in code:
                    print(f"Code snippet {i}: FAILED, '_eval_fn_by_expr_type' not defined.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # 2) Dynamic execution and logic testing
                exec_globals = {'Any': Any}  
                exec_locals = {}

                try:
                    # Dynamically execute snippet
                    exec(code, exec_globals, exec_locals)

                    # Initialize the class object if present
                    class_obj_name = [name for name in exec_locals if hasattr(exec_locals[name], '__init__')]
                    if not class_obj_name:
                        print(f"Code snippet {i}: FAILED, no class with '__init__' method found.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    class_instance = exec_locals[class_obj_name[0]]()

                    # Check if _eval_fn_by_expr_type is properly set as a dictionary
                    self.assertIsInstance(
                        class_instance._eval_fn_by_expr_type,
                        dict,
                        f"Code snippet {i} did not set '_eval_fn_by_expr_type' as a dictionary.",
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

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "__init__"
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