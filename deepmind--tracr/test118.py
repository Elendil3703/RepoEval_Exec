import unittest
import json
import os
from typing import Union

TEST_RESULT_JSONL = "test_result.jsonl"

class SequenceMap:
    def __init__(self, func, *sequences):
        self.func = func
        self.sequences = sequences

    def execute(self):
        return [self.func(*items) for items in zip(*self.sequences)]

class Map:
    def __init__(self, func, sequence):
        self.func = func
        self.sequence = sequence

    def execute(self):
        return [self.func(item) for item in self.sequence]

class SOp:
    def __init__(self, values):
        self.values = values

    def __radd__(self, other: Union["SOp", int]) -> "SOp":
        """other + self."""
        if isinstance(other, SOp):
            return SequenceMap(lambda x, y: x + y, other, self)
        return Map(lambda x: other + x, self)

    def execute(self):
        return self.values 

class TestSOpRadd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[117]  # Get the 118th JSON element

    def test_radd_snippets(self):
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Testing code snippet {i}...")

                # Define the test values
                test_values_self = SOp([1, 2, 3])
                test_values_other_sop = SOp([10, 20, 30])
                test_values_other_value = 5

                exec_globals = {
                    'SOp': SOp,
                    'SequenceMap': SequenceMap,
                    'Map': Map,
                    'Union': Union
                }
                exec_locals = {}

                try:
                    # Execute the provided user snippet
                    exec(code, exec_globals, exec_locals)

                    # Prepare the test instance of SOp
                    sop_instance = exec_locals.get('sop_instance', None)
                    if sop_instance is None:
                        sop_instance = SOp([1, 2, 3])

                    # Check when other is an SOp instance
                    result_sequence_map = sop_instance.__radd__(test_values_other_sop)
                    expected_sequence_map = SequenceMap(lambda x, y: x + y, test_values_other_sop, sop_instance)
                    self.assertEqual(result_sequence_map.execute(), expected_sequence_map.execute())

                    # Check when other is a value
                    result_map = sop_instance.__radd__(test_values_other_value)
                    expected_map = Map(lambda x: test_values_other_value + x, sop_instance)
                    self.assertEqual(result_map.execute(), expected_map.execute())

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__radd__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__radd__",
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
            if rec.get("function_name") != "__radd__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()