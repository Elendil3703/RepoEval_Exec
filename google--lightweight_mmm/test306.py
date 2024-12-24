import unittest
import json
import sys
import os
import re
from typing import Any  # Ensure Any is injected into the environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxMockFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[305]  # Get the 306th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 306th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Static check: Ensure code contains mock_model_function definition
                if "def mock_model_function" not in code:
                    print(f"Code snippet {i}: FAILED, 'def mock_model_function' not found.\n")
                    failed_count += 1
                    # Write failed record
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+mock_model_function\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'mock_model_function'.\n")
                    failed_count += 1
                    # Write failed record
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and logic testing
                exec_globals = {
                    'numpyro': type('Obj', (object,), {'deterministic': lambda *args, **kwargs: None}),
                    'trend': type('Obj', (object,), {'trend_with_exponent': lambda **kwargs: "mock_trend"}),
                    'Any': Any,  # Inject Any
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if mock_model_function exists
                    if 'mock_model_function' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'mock_model_function' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "mock_model_function",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test mock_model_function with a sample input
                    exec_locals['mock_model_function'](data="sample_data")

                    # Mock check
                    self.assertTrue(True, "Mock check passed without raising an error.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
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

        # Remove old records with function_name == "mock_model_function"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "mock_model_function"
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