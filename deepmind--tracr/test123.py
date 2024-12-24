import unittest
import json
import os
from typing import Union

TEST_RESULT_JSONL = "test_result.jsonl"


class MockSequenceMap:
    def __init__(self, func, x, y):
        self.func = func
        self.x = x
        self.y = y


class MockMap:
    def __init__(self, func, x):
        self.func = func
        self.x = x


class SOp:
    def __init__(self, value):
        self.value = value

    def __truediv__(self, other: Union["SOp", float, int]) -> "SOp":
        """self / other."""
        if isinstance(other, SOp):
            return MockSequenceMap(lambda x, y: x / y, self, other)
        return MockMap(lambda x: x / other, self)


class TestSOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Read the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[122]  # Get the 123rd JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 123rd JSON array")

    def test_division(self):
        """Dynamically test all code snippets in the JSON."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                exec_globals = {'MockSequenceMap': MockSequenceMap, 'MockMap': MockMap, 'SOp': SOp}
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet to test
                    exec(code, exec_globals, exec_locals)
                    sop_instance = exec_locals['SOp'](10)

                    # Perform operations and assert expectations
                    result_with_sop = sop_instance / exec_locals['SOp'](2)
                    self.assertIsInstance(
                        result_with_sop,
                        MockSequenceMap,
                        f"Code snippet {i} did not produce a MockSequenceMap instance.",
                    )

                    result_with_numeric = sop_instance / 2
                    self.assertIsInstance(
                        result_with_numeric,
                        MockMap,
                        f"Code snippet {i} did not produce a MockMap instance.",
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__truediv__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__truediv__",
                        "code": code,
                        "result": "failed"
                    })

        # Final test summary
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

        # Remove old records for __truediv__
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "__truediv__"]

        # Append new results
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()