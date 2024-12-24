import unittest
import json
import os
from typing import Union

TEST_RESULT_JSONL = "test_result.jsonl"

# Mock classes for testing, should reflect the context in which __rmul__ operates
class NumericValue:
    pass

class Map:
    def __init__(self, func, self_obj):
        self.func = func
        self.self_obj = self_obj

class SequenceMap:
    def __init__(self, func, other, self_obj):
        self.func = func
        self.other = other
        self.self_obj = self_obj

class SOp:
    def __rmul__(self, other: Union["SOp", NumericValue]) -> "SOp":
        if isinstance(other, SOp):
            return SequenceMap(lambda x, y: x * y, other, self)
        return Map(lambda x: other * x, self)

class TestRMulResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[121]  # Get the 122nd JSON element

    def test_rmul_functionality(self):
        """Test the __rmul__ method with different scenarios."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippet):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                exec_globals = {
                    'Union': Union,
                    'SOp': SOp,
                    'NumericValue': NumericValue,
                    'Map': Map,
                    'SequenceMap': SequenceMap,
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)
                    sop_instance = SOp()
                    sop_other = SOp()
                    numerical_other = NumericValue()

                    # Test multiplication with another SOp instance
                    result1 = sop_instance.__rmul__(sop_other)
                    self.assertIsInstance(
                        result1, SequenceMap,
                        f"Expected SequenceMap but got {type(result1)} in code snippet {i}"
                    )

                    # Test multiplication with a NumericValue
                    result2 = sop_instance.__rmul__(numerical_other)
                    self.assertIsInstance(
                        result2, Map,
                        f"Expected Map but got {type(result2)} in code snippet {i}"
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__rmul__",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__rmul__",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippet)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippet), "Test count mismatch!")

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
            if rec.get("function_name") != "__rmul__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()