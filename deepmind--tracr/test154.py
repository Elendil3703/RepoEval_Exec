import unittest
import json
import os
from typing import List, Optional

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCustomAssertions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[153]  # Get the 154th JSON element (index 153)
        if not cls.code_snippet:
            raise ValueError("No code snippet found at index 153")

    def test_assert_sequence_equal_when_expected_is_not_none(self):
        """Test the assertSequenceEqualWhenExpectedIsNotNone function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results for JSONL

        global_namespace = {}
        # Load the code snippet to have the tested function available
        exec(self.code_snippet, global_namespace)

        test_cases = [
            {"actual": [1, 2, 3], "expected": [1, 2, 3], "should_fail": False},
            {"actual": [1, 2, None], "expected": [1, 2, 3], "should_fail": True},
            {"actual": [1, None, 3], "expected": [1, 4, 3], "should_fail": False},
            {"actual": [None, 2, 3], "expected": [0, 2, 3], "should_fail": False},
            {"actual": [1, 2, 3], "expected": [1, 2, None], "should_fail": False},
        ]

        for i, test_case in enumerate(test_cases):
            with self.subTest(test_index=i):
                actual_seq = test_case["actual"]
                expected_seq = test_case["expected"]
                should_fail = test_case["should_fail"]

                # Create an instance of the class that has the method
                test_instance = self.__class__('run')
                test_instance.longMessage = True

                method = global_namespace['assertSequenceEqualWhenExpectedIsNotNone']
                method_bound = method.__get__(test_instance)

                try:
                    method_bound(actual_seq, expected_seq)
                    
                    if should_fail:
                        print(f"Test case {i}: FAILED, expected failure but it passed.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "assertSequenceEqualWhenExpectedIsNotNone",
                            "actual": actual_seq,
                            "expected": expected_seq,
                            "result": "failed"
                        })
                    else:
                        print(f"Test case {i}: PASSED as expected.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "assertSequenceEqualWhenExpectedIsNotNone",
                            "actual": actual_seq,
                            "expected": expected_seq,
                            "result": "passed"
                        })
                except AssertionError as e:
                    if should_fail:
                        print(f"Test case {i}: FAILED as expected.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "assertSequenceEqualWhenExpectedIsNotNone",
                            "actual": actual_seq,
                            "expected": expected_seq,
                            "result": "passed"
                        })
                    else:
                        print(f"Test case {i}: FAILED unexpectedly with error: {e}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "assertSequenceEqualWhenExpectedIsNotNone",
                            "actual": actual_seq,
                            "expected": expected_seq,
                            "result": "failed"
                        })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "assertSequenceEqualWhenExpectedIsNotNone"
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