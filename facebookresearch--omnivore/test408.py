import unittest
import json
import os
import re
from typing import Union, Set
from fnmatch import filter
import logging
from omegaconf import DictConfig
import torch.nn as nn

TEST_RESULT_JSONL = "test_result.jsonl"

class TestUnixParamPatternToParameterNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and get the specific code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[407]  # Get the 408th JSON element (index 407)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the 408th JSON array")
    
    def test_unix_param_pattern_to_parameter_names(self):
        """Test the unix_param_pattern_to_parameter_names function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into JSONL

        # Extract the function code
        code = self.code_snippet

        func_pattern = r"def\s+unix_param_pattern_to_parameter_names\s*\("
        if not re.search(func_pattern, code):
            print("Code snippet: FAILED, incorrect signature for 'unix_param_pattern_to_parameter_names'.\n")
            failed_count += 1
            results.append({
                "function_name": "unix_param_pattern_to_parameter_names",
                "code": code,
                "result": "failed"
            })
        else:
            exec_globals = {
                'DictConfig': DictConfig,
                'nn': nn,
                'Union': Union,
                'Set': Set,
                'fnmatch': filter,
                'logging': logging,
            }
            exec_locals = {}

            try:
                # Dynamically execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Test inputs and expected outputs
                test_cases = [
                    (
                        DictConfig({"param_names": ["weight", "bias"]}),
                        nn.Linear(10, 5),
                        {"weight", "bias"}
                    ),
                    (
                        DictConfig({"param_names": ["*weight"]}),
                        nn.Linear(10, 5),
                        {"weight"}
                    ),
                    (
                        DictConfig({"param_names": ["nonexistent"]}),
                        nn.Linear(10, 5),
                        None
                    ),
                    (
                        DictConfig({}),
                        nn.Linear(10, 5),
                        set()
                    )
                ]

                for i, (scheduler_cfg, model, expected) in enumerate(test_cases):
                    with self.subTest(test_index=i):
                        fn = exec_locals['unix_param_pattern_to_parameter_names']
                        try:
                            result = fn(scheduler_cfg, model)
                            self.assertEqual(result, expected)
                            print(f"Test case {i}: PASSED")
                            passed_count += 1
                            results.append({
                                "function_name": "unix_param_pattern_to_parameter_names",
                                "code": code,
                                "result": "passed"
                            })
                        except AssertionError as e:
                            print(f"Test case {i}: FAILED with AssertionError: {e}")
                            failed_count += 1
                            results.append({
                                "function_name": "unix_param_pattern_to_parameter_names",
                                "code": code,
                                "result": "failed"
                            })
            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "unix_param_pattern_to_parameter_names",
                    "code": code,
                    "result": "failed"
                })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")
        
        # Write test results into test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the same function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "unix_param_pattern_to_parameter_names"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()