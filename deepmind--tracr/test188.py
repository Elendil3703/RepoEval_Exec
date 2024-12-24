import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestOvFunResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        
        # Get the 188th code element (index 187)
        cls.code_snippet = data[187]

        if not cls.code_snippet:
            raise ValueError("Expected non-empty code snippet at index 187")

    def test_ov_fun_snippet(self):
        """Test the specific `ov_fun` implementation."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet

        exec_globals = {
            'bases': Any,  # example injection for missing names
            'use_bos_for_default_output': True,
            'input_dir': 'bos_direction',
            'bos_direction': 'bos_direction',
            'default_output': 'default_output',
            'output_space': Any,
            'value_to_output': {'input_dir': 'return_value'}
        }
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)

            # Check if `ov_fun` exists in the executed locals
            if 'ov_fun' not in exec_locals:
                print(f"Code snippet failed: 'ov_fun' not found.\n")
                failed_count += 1
                results.append({
                    "function_name": "ov_fun",
                    "code": code,
                    "result": "failed"
                })
                return

            # Access the `ov_fun` for testing
            ov_fun_ref = exec_locals['ov_fun']

            # Test case: Check for default output
            output = ov_fun_ref('bos_direction')
            if output == 'default_output':
                passed_count += 1
                results.append({
                    "function_name": "ov_fun",
                    "code": code,
                    "result": "passed"
                })
            else:
                failed_count += 1
                results.append({
                    "function_name": "ov_fun",
                    "code": code,
                    "result": "failed"
                })
                
            # Test case: Check for value_to_output mapping
            exec_globals['use_bos_for_default_output'] = False
            output2 = ov_fun_ref('input_dir')
            if output2 == 'return_value':
                passed_count += 1
                results.append({
                    "function_name": "ov_fun",
                    "code": code,
                    "result": "passed"
                })
            else:
                failed_count += 1
                results.append({
                    "function_name": "ov_fun",
                    "code": code,
                    "result": "failed"
                })

            print(f"Code snippet PASSED tests.\n" if failed_count == 0 else f"Code snippet FAILED tests.\n")

        except Exception as e:
            print(f"Code snippet failed with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "ov_fun",
                "code": code,
                "result": "failed"
            })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")

        # Prepare and write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name "ov_fun"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "ov_fun"
        ]

        # Add new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()