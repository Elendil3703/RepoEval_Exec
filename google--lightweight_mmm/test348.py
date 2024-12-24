import unittest
import json
import sys
import re
import os
from unittest import mock

TEST_RESULT_JSONL = "test_result.jsonl"

class TestOptimizeMediaResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[347]  # Get the 348th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test setUp methods in the JSON code snippets."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Static Checks -------------------
                # Ensure 'setUp' function exists in the snippet
                if "def setUp" not in code:
                    print(f"Code snippet {i}: FAILED, 'def setUp' not found in code.\n")
                    failed_count += 1
                    # Record failure
                    results.append({
                        "function_name": "setUp",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Check for patching logic in setUp
                if "mock.patch.object" not in code:
                    print(f"Code snippet {i}: FAILED, 'mock.patch.object' not found.\n")
                    failed_count += 1
                    # Record failure
                    results.append({
                        "function_name": "setUp",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'mock': mock,
                    'object': object
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Mock object to test if patched method is called
                    class MockOptimize:
                        def minimize(self):
                            pass

                    class TestClass:
                        def setUp(self):
                            super().setUp()
                            self.mock_minimize = mock.patch.object(
                                MockOptimize, "minimize", autospec=True)
                        
                        def test_minimize_called(self):
                            instance = MockOptimize()
                            instance.minimize()
                            # Assert if the minimize method was called
                            self.mock_minimize.assert_called_once_with()

                    # Instantiate and run the test
                    test_instance = TestClass()
                    test_instance.setUp()
                    test_instance.test_minimize_called()

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "setUp",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "setUp",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
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

        # Delete old records with function_name == "setUp"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "setUp"
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