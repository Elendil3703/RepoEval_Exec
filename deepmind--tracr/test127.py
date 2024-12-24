import unittest
import json
import sys
import re
import os
from typing import Union, Any

TEST_RESULT_JSONL = "test_result.jsonl"


class SOp:
    pass


class SequenceMap(SOp):
    def __init__(self, func, x, y):
        self.func = func
        self.x = x
        self.y = y

    def __call__(self):
        return self.func(self.x, self.y)


class Map(SOp):
    def __init__(self, func, x):
        self.func = func
        self.x = x

    def __call__(self):
        return self.func(self.x)


class TestGroundTruthFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[126]  # Get the required JSON element
        if not cls.code_snippet:
            raise ValueError("Expected code snippet not found in position 126")

    def test_ground_truth_function(self):
        """Dynamically test the ground truth function __ror__."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []  # Collect test results to be written to JSONL

        code = self.__class__.code_snippet

        print("Running test...")
        exec_globals = {
            'SOp': SOp,
            'SequenceMap': SequenceMap,
            'Map': Map,
            'Union': Union,
            'Any': Any,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if __ror__ method is defined
            if '__ror__' not in exec_locals:
                print(f"__ror__: FAILED, '__ror__' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "__ror__",
                    "code": code,
                    "result": "failed"
                })
            else:
                # Test the __ror__ functionality with some crafted examples
                class TestSOp(SOp):
                    def __ror__(self, other):
                        return exec_locals['__ror__'](self, other)

                sop_instance = TestSOp()

                # Test with SOp instance
                sop_other = SOp()
                result = sop_other | sop_instance
                self.assertIsInstance(
                    result, SequenceMap,
                    f"Expected SequenceMap but got {type(result)}"
                )

                # Test with a numeric-like value (assuming NumericValue as a number)
                numeric_other = 5
                result = numeric_other | sop_instance
                self.assertIsInstance(
                    result, Map,
                    f"Expected Map but got {type(result)}"
                )

                print(f"__ror__: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "__ror__",
                    "code": code,
                    "result": "passed"
                })

        except Exception as e:
            print(f"__ror__: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__ror__",
                "code": code,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name == "__ror__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__ror__"
        ]

        # Extend existing records with new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()