import unittest
import json
import os
from typing import Callable, Dict, Tuple, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroupDictByKey(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[95]  # Get the 96th JSON element

        if not cls.code_snippet:
            raise ValueError("Expected code snippet at index 95")

    def test_group_dict_by_key(self):
        """Test the group_dict_by_key function with multiple scenarios."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        # Insert the code snippet in the current namespace
        exec_globals = {"Any": Any}  # Inject Any if needed
        exec(self.code_snippet, exec_globals)

        # Ensure function is loaded
        if "group_dict_by_key" not in exec_globals:
            raise ImportError("group_dict_by_key function is not defined after exec")

        group_dict_by_key = exec_globals["group_dict_by_key"]

        # Define test cases
        test_cases = [
            {"cond": lambda x: x.startswith('a'), "dict": {"apple": 1, "banana": 2}, "expected": ({"apple": 1}, {"banana": 2})},
            {"cond": lambda x: len(x) == 4,     "dict": {"pear": 1, "peach": 3},  "expected": ({"pear": 1}, {"peach": 3})}
        ]

        for i, test in enumerate(test_cases):
            with self.subTest(test_index=i):
                try:
                    print(f"Running test case {i}...")
                    result = group_dict_by_key(test["cond"], test["dict"])
                    self.assertEqual(
                        result,
                        test["expected"],
                        f"Test case {i} failed: expected {test['expected']}, got {result}"
                    )
                    print(f"Test case {i}: PASSED\n")
                    passed_count += 1
                    results.append({
                        "function_name": "group_dict_by_key",
                        "test_case": test,
                        "result": "passed"
                    })
                except AssertionError as ae:
                    print(f"Test case {i}: FAILED, {ae}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "group_dict_by_key",
                        "test_case": test,
                        "result": "failed"
                    })

        # Final summary
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

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "group_dict_by_key"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()