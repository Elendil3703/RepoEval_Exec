import unittest
import json
import os
import fnmatch
from typing import List, Set, Union

TEST_RESULT_JSONL = "test_result.jsonl"

class TestUnixPatternToParameterNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[391]  # Get the 392nd JSON element (zero-indexed)

    def test_unix_pattern_to_parameter_names(self):
        """Test _unix_pattern_to_parameter_names function."""
        results = []  # Collect results to be written into JSONL

        for i, code in enumerate(self.code_snippet):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                func_name = "_unix_pattern_to_parameter_names"
                
                if func_name not in code:
                    print(f"Code snippet {i}: FAILED, '{func_name}' not found in code.\n")
                    results.append({
                        "function_name": func_name,
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Inject required functions and variables
                exec_globals = {
                    'fnmatch': fnmatch,
                    'List': List,
                    'Set': Set,
                    'Union': Union,
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if the function is defined
                    if func_name not in exec_locals:
                        raise ValueError(f"Function '{func_name}' is not defined in the code.")

                    # Retrieve the function
                    unix_pattern_func = exec_locals[func_name]

                    # Define test examples
                    test_cases = [
                        (['param*'], {'param1', 'param2', 'different'}, {'param1', 'param2'}),
                        (['*diff*'], {'param1', 'different', 'diff_param'}, {'different', 'diff_param'}),
                        (['*'], {'param1', 'param2'}, {'param1', 'param2'}),
                        (['unknown'], {'param1', 'param2'}, ValueError),
                    ]

                    for constraints, all_params, expected in test_cases:
                        if isinstance(expected, type) and issubclass(expected, Exception):
                            with self.assertRaises(expected):
                                unix_pattern_func(constraints, all_params)
                        else:
                            result = unix_pattern_func(constraints, all_params)
                            self.assertEqual(result, expected)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    results.append({
                        "function_name": func_name,
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    results.append({
                        "function_name": func_name,
                        "code": code,
                        "result": "failed"
                    })

        # Write results to JSONL file
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
            if rec.get("function_name") != "_unix_pattern_to_parameter_names"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()