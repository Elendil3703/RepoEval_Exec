import unittest
import json
import sys
import re
import os
from typing import Any  # Ensure Any is available in the injected environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[33]  # Get the 34th JSON element (index 33)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static checks -------------------
                if "__init__" not in code:
                    print(f"Code snippet {i}: FAILED, '__init__' not found in code.\n")
                    failed_count += 1
                    # Record failure
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                init_pattern = r"def\s+__init__\s*\("
                if not re.search(init_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for '__init__'.\n")
                    failed_count += 1
                    # Record failure
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution and testing -------------------
                exec_globals = {
                    'super': super,
                    'hf_get_decoder': lambda model: model,  # Mock function
                    'transformers': type('transformers', (), {
                        'PreTrainedModel': type('PreTrainedModel', (), {})
                    }),
                    'Any': Any,  # Inject Any
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if __init__ really exists within the executed class
                    found_class = None
                    for name, cls in exec_locals.items():
                        if isinstance(cls, type):
                            if '__init__' in cls.__dict__:
                                found_class = cls
                                break

                    if not found_class:
                        print(f"Code snippet {i}: FAILED, '__init__' not found in the executed class.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Simulate a test case to initialize the class
                    mock_model = exec_globals['transformers'].PreTrainedModel()
                    instance = found_class(mock_model, num_layers_unfrozen=2)

                    # Check attributes set in the __init__ function
                    self.assertTrue(hasattr(instance, 'dropout'), f"Code snippet {i}: 'dropout' attribute was not set.")
                    self.assertTrue(hasattr(instance, 'is_decoder') and instance.is_decoder,
                                    f"Code snippet {i}: 'is_decoder' attribute was incorrectly set or missing.")

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

        # Summary of tests
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if not exist)
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