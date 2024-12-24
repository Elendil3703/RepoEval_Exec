import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class MockSearchIndex:
    def __init__(self):
         self.reset_called = False

    def reset(self):
        self.reset_called = True

class TestResetIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[375]  # Get the specific JSON element
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
                
                # ------------------- Dynamic Execution and Logic Testing -------------------
                exec_globals = {
                    'Any': Any,  # Inject Any
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if reset_index exists
                    if 'reset_index' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'reset_index' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "reset_index",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Testing logic for reset_index
                    class SomeClass:
                        def __init__(self):
                            self.search_index = MockSearchIndex()

                        def reset_index(self):
                            exec_locals['reset_index'](self)

                    # Initialize class and call method
                    obj = SomeClass()
                    obj.reset_index()

                    # Assertions to verify behavior
                    self.assertIsNone(
                        obj.search_index,
                        f"Code snippet {i} did not set search_index to None after reset."
                    )
                    self.assertTrue(
                        obj.search_index.reset_called,
                        f"Code snippet {i} did not call reset on search_index."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "reset_index",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "reset_index",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "reset_index"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "reset_index"
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