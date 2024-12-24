import unittest
import json
import os
from typing import List, Tuple

TEST_RESULT_JSONL = "test_result.jsonl"

def fill_by_nines(number: int, nines_count: int) -> int:
    """Fills the last n digits of a number with nines."""
    str_num = str(number)
    if len(str_num) < nines_count:
        return int("9" * nines_count)
    return int(str_num[:-nines_count] + "9" * nines_count)

def fill_by_zeros(number: int, zeros_count: int) -> int:
    """Fills the last n digits of a number with zeros."""
    str_num = str(number)
    if len(str_num) < zeros_count:
        return 0
    return int(str_num[:-zeros_count] + "0" * zeros_count)


class TestSplitToRanges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[25]  # Get the 26th JSON element (0-indexed)

    def test_split_to_ranges(self):
        """Test the split_to_ranges functionality with various inputs."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        # Define test cases
        test_cases: List[Tuple[int, int, List[int]]] = [
            (1, 10, [1, 9, 10]),
            (100, 500, [100, 199, 299, 399, 499, 500]),
            (555, 555, [555]),
            (990, 1000, [990, 999, 1000]),
            (0, 9999, [0, 9, 99, 999, 9999]),
            (4500, 4570, [4500, 4509, 4519, 4529, 4539, 4549, 4559, 4569, 4570]),
        ]

        for i, (min_, max_, expected) in enumerate(test_cases):
            with self.subTest(min_=min_, max_=max_):
                exec_globals = {'fill_by_nines': fill_by_nines, 'fill_by_zeros': fill_by_zeros}
                exec_locals = {}
                exec(self.code_snippet, exec_globals, exec_locals)

                try:
                    split_to_ranges = exec_locals['split_to_ranges']
                    result = split_to_ranges(min_, max_)
                    self.assertEqual(
                        result, expected,
                        f"Test {i} failed for min_={min_}, max_={max_}: expected {expected} but got {result}."
                    )
                    passed_count += 1
                    results.append({
                        "function_name": "split_to_ranges",
                        "input": {"min_": min_, "max_": max_},
                        "expected_output": expected,
                        "actual_output": result,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Test {i} for min_={min_}, max_={max_} failed with error: {e}")
                    failed_count += 1
                    results.append({
                        "function_name": "split_to_ranges",
                        "input": {"min_": min_, "max_": max_},
                        "expected_output": expected,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if nonexistent)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "split_to_ranges"
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "split_to_ranges"]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()