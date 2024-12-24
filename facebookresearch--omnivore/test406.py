import unittest
import json
import os
import re
from typing import Union, Set, Dict
from unittest.mock import MagicMock

TEST_RESULT_JSONL = "test_result.jsonl"

class TestUnixPatternToParameterNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[405]  # Get the 406th JSON element (index 405)

    def test_unix_pattern_to_parameter_names(self):
        """Test the unix_pattern_to_parameter_names function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        exec_globals = {
            'DictConfig': Dict,
            'nn': MagicMock(),  # Mock nn.Module
            'Union': Union,
            'Set': Set,
            'unix_param_pattern_to_parameter_names': MagicMock(return_value={"param1", "param2"}),
            'unix_module_cls_pattern_to_parameter_names': MagicMock(return_value={"param3"}),
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            if 'unix_pattern_to_parameter_names' not in exec_locals:
                raise ValueError("Function 'unix_pattern_to_parameter_names' not defined in the snippet.")

            # Extract the function to test
            func = exec_locals['unix_pattern_to_parameter_names']

            # Mock scheduler_cfg
            scheduler_cfg = {
                'param_names': ['pattern1'],
                'module_cls_names': ['pattern2']
            }

            # Create a MagicMock model
            model = MagicMock()

            # Run the function
            result = func(scheduler_cfg, model)

            # Perform assertions
            expected_result = {"param1", "param2", "param3"}

            self.assertIsInstance(result, set, "Returned value is not a set.")
            self.assertEqual(result, expected_result, "Incorrect set of parameter names returned.")

            passed_count += 1
            results.append({
                "function_name": "unix_pattern_to_parameter_names",
                "code": self.code_snippet,
                "result": "passed"
            })
            print("Code snippet: PASSED all assertions.\n")
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "unix_pattern_to_parameter_names",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Writing results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records related to function_name "unix_pattern_to_parameter_names"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "unix_pattern_to_parameter_names"
        ]

        # Add new results
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()