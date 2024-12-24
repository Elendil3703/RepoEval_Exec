import unittest
import json
import sys
import re
import os
from typing import List, Optional

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[394]  # Get the 395th JSON element (index 394)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the selected JSON array")

    def test_init_method(self):
        """Test all code snippets for the __init__ method presence and functionality."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # 1) Static check: Ensure __init__ is defined
                if "def __init__" not in code:
                    print(f"Code snippet {i}: FAILED, '__init__' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # 2) Regex to find the __init__ signature
                init_pattern = r"def\s+__init__\s*\(\s*self\s*,\s*source_pattern\s*:\s*str\s*,\s*target_patterns\s*:\s*List\[str\]\s*(,\s*key_pattern\s*:\s*Optional\[List\[str\]\]\s*=\s*None\s*)?\)"
                if not re.search(init_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for '__init__'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue


                # 3) Dynamic execution and testing
                exec_globals = {
                    'List': List,
                    'Optional': Optional
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    # Check if the class is defined in the local scope
                    class_candidates = [cls_name for cls_name, cls_obj in exec_locals.items() if isinstance(cls_obj, type)]
                    if not class_candidates:
                        print(f"Code snippet {i}: FAILED, no class with '__init__' found.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    cls_name = class_candidates[0]
                    cls_obj = exec_locals[cls_name]

                    # Create an instance of the class to test the __init__ functionality
                    instance = cls_obj("source", ["target1", "target2"], ["key1", "key2"])

                    self.assertEqual(instance.source_pattern, "source", f"Code snippet {i}: FAILED, source_pattern mismatch.")
                    self.assertEqual(instance.target_patterns, ["target1", "target2"], f"Code snippet {i}: FAILED, target_patterns mismatch.")
                    self.assertEqual(instance.key_pattern, ["key1", "key2"], f"Code snippet {i}: FAILED, key_pattern mismatch.")

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
            if rec.get("function_name") != "__init__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()