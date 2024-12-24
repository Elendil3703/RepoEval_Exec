import unittest
import json
import sys
import re
import os
from typing import Any, Callable, Dict, Tuple, List  # 确保注入的环境中有 Any, Callable, Dict, Tuple, List

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroupByPrefixAndTrimResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and read the specific test case
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[96]  # Get the 97th JSON element (index 96)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet in the 97th JSON array")

    def test_groupby_prefix_and_trim(self):
        """Test the groupby_prefix_and_trim function from the code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into JSONL

        # ------------------ Static Checks ------------------
        if "groupby_prefix_and_trim" not in self.code_snippet:
            print("Code snippet: FAILED, 'groupby_prefix_and_trim' not found in code.\n")
            failed_count += 1
            results.append({
                "function_name": "groupby_prefix_and_trim",
                "code": self.code_snippet,
                "result": "failed"
            })
            return

        # ------------------ Dynamic Execution ------------------
        exec_globals = {
            'Any': Any,
            'Callable': Callable,
            'Dict': Dict,
            'Tuple': Tuple,
            'List': List,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check if groupby_prefix_and_trim exists
            if 'groupby_prefix_and_trim' not in exec_locals:
                print("Code snippet: FAILED, 'groupby_prefix_and_trim' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "groupby_prefix_and_trim",
                    "code": self.code_snippet,
                    "result": "failed"
                })
                return

            # Define a mock for group_dict_by_key and partial
            def mock_group_dict_by_key(func, d):
                return {k: v for k, v in d.items() if func(k)}, {k: v for k, v in d.items() if not func(k)}

            def mock_partial(func, prefix):
                def wrapped(key):
                    return func(prefix, key)
                return wrapped

            # Override exec_globals to include mocks
            exec_globals['group_dict_by_key'] = mock_group_dict_by_key
            exec_globals['partial'] = mock_partial

            # Test data
            test_dict = {
                'prefix_key1': 'value1',
                'key2': 'value2',
                'prefix_key3': 'value3',
                'key4': 'value4'
            }
            expected_with_prefix = {'key1': 'value1', 'key3': 'value3'}
            expected_without_prefix = {'key2': 'value2', 'key4': 'value4'}

            # Call the function with test data
            fn = exec_locals['groupby_prefix_and_trim']
            result_with_prefix, result_without_prefix = fn('prefix_', test_dict)

            # Validate the function's output
            self.assertEqual(result_with_prefix, expected_with_prefix, "Result does not match expected with prefix.")
            self.assertEqual(result_without_prefix, expected_without_prefix, "Result does not match expected without prefix.")

            print("Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "groupby_prefix_and_trim",
                "code": self.code_snippet,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "groupby_prefix_and_trim",
                "code": self.code_snippet,
                "result": "failed"
            })

        # ============= Write results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if not exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for groupby_prefix_and_trim
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "groupby_prefix_and_trim"
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