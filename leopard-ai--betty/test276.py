import unittest
import json
import os
from typing import Optional, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestNegWithNone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[275]  # Get the 276th JSON element (index 275)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 276th JSON array")

    def test_neg_with_none(self):
        """Dynamically test all code snippets in the JSON for neg_with_none function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect the test results to write to JSONL

        # Define test cases
        test_cases = [
            (None, None),
            (5, -5),
            (-3, 3),
            (0, 0),
            (10.5, -10.5)
        ]

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static Check -------------------
                if "def neg_with_none" not in code:
                    print(f"Code snippet {i}: FAILED, function 'neg_with_none' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "neg_with_none",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution -------------------
                exec_globals = {
                    'Optional': Optional,
                    'Any': Any
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if neg_with_none is actually defined
                    if 'neg_with_none' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'neg_with_none' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "neg_with_none",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Verify neg_with_none logic
                    func = exec_locals['neg_with_none']
                    for input_value, expected_output in test_cases:
                        with self.subTest(input_value=input_value):
                            result = func(input_value)
                            self.assertEqual(result, expected_output,
                                             f"Code snippet {i}: Expected {expected_output} for input {input_value}, got {result}.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "neg_with_none",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "neg_with_none",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if not exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name == "neg_with_none"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "neg_with_none"
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