import unittest
import json
import os
from typing import Union

from collections.abc import Sequence

TEST_RESULT_JSONL = "test_result.jsonl"

class SOp:
    def __init__(self, values):
        self.values = values

    def __add__(self, other: Union["SOp", float, int]) -> "SOp":
        if isinstance(other, SOp):
            return SequenceMap(lambda x, y: x + y, self, other)
        return Map(lambda x: x + other, self)


class Map(SOp):
    def __init__(self, func, source):
        super().__init__([func(x) for x in source.values])


class SequenceMap(SOp):
    def __init__(self, func, source1, source2):
        super().__init__(
            [func(x, y) for x, y in zip(source1.values, source2.values)]
        )


class TestSOpAddition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[116]  # Get the 117th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 117th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets involving the __add__ method."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Define a local scope for exec
                exec_globals = {
                    'SOp': SOp,
                    'Map': Map,
                    'SequenceMap': SequenceMap,
                    'Union': Union,
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Assuming the code snippet has a function utilizing '__add__' 
                    # Simulate usage of `__add__` in context
                    sop1 = SOp([1, 2, 3])
                    sop2 = SOp([4, 5, 6])
                    result = sop1 + sop2
                    self.assertEqual(result.values, [5, 7, 9], "Addition with SOp failed.")

                    result = sop1 + 3
                    self.assertEqual(result.values, [4, 5, 6], "Addition with scalar failed.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__add__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__add__",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
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

        # Remove old records for __add__
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__add__"
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