import unittest
import json
import os
import pandas as pd
from typing import Any, Union, List

# Ensure the name is consistent with the instruction
TEST_RESULT_JSONL = "test_result.jsonl"

class PandasDataFrame(pd.DataFrame):
    pass

class LazyColumns(list):
    pass

def _check_type(columns: Any, expected_types: set):
    if not isinstance(columns, tuple(expected_types)):
        raise TypeError("Invalid type for columns")

class TestSelectFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[439]  # Get the 440th JSON element (index 439)

        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the specified JSON array")

    def test_select_function(self):
        """Dynamically test 'select' function in the JSON code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Environment for code execution
                exec_globals = {
                    'PandasDataFrame': PandasDataFrame,
                    'LazyColumns': LazyColumns,
                    '_check_type': _check_type,
                }
                exec_locals = {}

                try:
                    # Execute code snippet to define the select function
                    exec(code, exec_globals, exec_locals)

                    # Check if select function is defined
                    if 'select' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'select' function not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "select",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    select = exec_locals['select']

                    # Prepare test cases
                    df = PandasDataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
                    
                    # Test different column selections
                    try:
                        # Valid selection with list
                        result = select(df, LazyColumns(['a', 'b']))
                        self.assertEqual(list(result.columns), ['a', 'b'])

                        # Valid selection with string
                        result = select(df, 'a')
                        self.assertEqual(list(result.columns), ['a'])

                        # Invalid selection, non-unique
                        with self.assertRaises(KeyError):
                            select(df, ['a', 'a'])

                        # Invalid selection, column does not exist
                        with self.assertRaises(KeyError):
                            select(df, 'd')

                        print(f"Code snippet {i}: PASSED all assertions.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "select",
                            "code": code,
                            "result": "passed"
                        })
                    except Exception as e:
                        print(f"Code snippet {i}: FAILED during runtime tests with error: {e}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "select",
                            "code": code,
                            "result": "failed"
                        })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "select",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Append results to the test_result.jsonl
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
            if rec.get("function_name") != "select"
        ]

        # Extend with new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()