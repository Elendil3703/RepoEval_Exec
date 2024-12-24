import unittest
import json
import os
from typing import Union

TEST_RESULT_JSONL = "test_result.jsonl"

class SequenceMap:
    def __init__(self, func, *args):
        self.func = func
        self.args = args

class Map:
    def __init__(self, func, *args):
        self.func = func
        self.args = args

class SOp:
    # Placeholder for the SOp class
    pass

class NumericValue:
    # Placeholder for the NumericValue class to simulate Union type hinting
    pass

class TestOrOperation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[124]  # Get the 125th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 125th JSON array")

    def test_or_operation(self):
        """Test __or__ function in code snippets with specific cases."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "__or__" not in code:
                    print(f"Code snippet {i}: FAILED, '__or__' method not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__or__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'SOp': SOp,
                    'SequenceMap': SequenceMap,
                    'Map': Map,
                    'Union': Union,
                    'NumericValue': NumericValue
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    self.assertTrue('SOp' in exec_locals, "SOp class not properly defined in exec_locals.")

                    test_instance = SOp()
                    test_other_sop = SOp()
                    test_numeric_value = 42

                    result_with_sop = test_instance.__or__(test_other_sop)
                    result_with_numeric = test_instance.__or__(test_numeric_value)

                    self.assertIsInstance(result_with_sop, SequenceMap, "Expected result to be a SequenceMap when 'other' is SOp.")
                    self.assertIsInstance(result_with_numeric, Map, "Expected result to be a Map when 'other' is NumericValue.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__or__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__or__",
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

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__or__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()