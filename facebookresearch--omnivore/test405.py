import unittest
import json
import os
import itertools
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class MockModel:
    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return ((name, param) for name, param in self._params.items())

class TestValidateParamGroupParams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and get the data at index 404
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[404]  # Get the 405th JSON element
        if not cls.code_snippet:
            raise ValueError("Expected at least one code snippet for the index 404")

    def test_validate_param_group_params(self):
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect the results to be written to JSONL

        # Dynamic execution
        exec_globals = {
            'mock_model': MockModel,
            'itertools': itertools,
        }
        exec_locals = {}

        try:
            exec(self.code_snippet, exec_globals, exec_locals)
            
            validate_func = exec_locals.get('validate_param_group_params', None)
            if not validate_func:
                raise ValueError("Function 'validate_param_group_params' not found")

            # Test cases
            param_groups = [
                {"params": ["param1", "param2"]},
                {"params": ["param3"]}
            ]
            model = MockModel({"param1": "value1", "param2": "value2", "param3": "value3"})
            
            # Execute function to test correctness
            validate_func(param_groups, model)

            print(f"Code snippet 404: PASSED basic test case.")
            passed_count += 1
            results.append({
                "function_name": "validate_param_group_params",
                "code": self.code_snippet,
                "result": "passed"
            })
            
        except Exception as e:
            print(f"Code snippet 404: FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "validate_param_group_params",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write results to test_result.jsonl
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
            if rec.get("function_name") != "validate_param_group_params"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()