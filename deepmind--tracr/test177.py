import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[176]  # Get the 177th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_make_sort_function(self):
        """Test the make_sort function in the provided code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "def make_sort" not in code:
                    print(f"Code snippet {i}: FAILED, 'make_sort' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "make_sort",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'rasp': RaspMock(),  # Mock the rasp module
                    'make_sort_unique': make_sort_unique_mock,  # Mock the make_sort_unique function
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if make_sort is defined
                    if 'make_sort' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'make_sort' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "make_sort",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Obtain the function
                    make_sort = exec_locals['make_sort']

                    # Mock data for testing
                    vals = [2, 4, 3, 1]
                    keys = [2, 4, 3, 1]
                    max_seq_len = 5
                    min_key = 1

                    # Execute the function and assert results
                    result = make_sort(vals, keys, max_seq_len=max_seq_len, min_key=min_key)
                    expected_result = [1, 2, 3, 4]  # Expected sorted result

                    # Perform assertion
                    self.assertEqual(result, expected_result, f"Code snippet {i} did not sort the list correctly.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "make_sort",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "make_sort",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
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

        # Remove old records for make_sort
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "make_sort"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

# Mock implementations to replace dependencies for testing
class RaspMock:
    def SequenceMap(self, func, keys, indices):
        # Generate modified keys as described
        return [func(x, i) for i, x in enumerate(keys)]

    @property
    def indices(self):
        return list(range(100))  # Mock of indices

def make_sort_unique_mock(vals, keys):
    # Mock sorting by keys
    return sorted(vals)

if __name__ == "__main__":
    unittest.main()