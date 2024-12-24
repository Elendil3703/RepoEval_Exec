import unittest
import json
import os
import sys
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"


class TestCheckTypeFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[451]  # Get the 452nd JSON element

    def test_code_snippets(self):
        """Dynamically test the code snippet for _check_type function"""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        code = self.code_snippets
        
        # Check if the expected function name is present in the code
        if "_check_type" not in code:
            print("Code snippet FAILED, '_check_type' not found in code.\n")
            failed_count += 1
            # Write failure record
            results.append({
                "function_name": "_check_type",
                "code": code,
                "result": "failed"
            })
        else:
            exec_globals = {
                'Any': Any,
            }
            exec_locals = {}

            try:
                # Dynamically execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Check if _check_type function is present
                if '_check_type' not in exec_locals:
                    print("Code snippet FAILED, '_check_type' not found in exec_locals.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_check_type",
                        "code": code,
                        "result": "failed"
                    })
                else:
                    # Test cases for _check_type function
                    _check_type = exec_locals['_check_type']
                    
                    # Valid cases
                    try:
                        _check_type(5, int)
                        _check_type(5, {int, None})
                        _check_type(None, {int, None})
                        
                        # Invalid cases
                        with self.assertRaises(TypeError):
                            _check_type(5, str)
                        
                        with self.assertRaises(TypeError):
                            _check_type(None, int)

                        print("Code snippet PASSED all assertions.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "_check_type",
                            "code": code,
                            "result": "passed"
                        })
                        
                    except Exception as e:
                        print(f"Code snippet FAILED with error: {e}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_check_type",
                            "code": code,
                            "result": "failed"
                        })
            except Exception as e:
                print(f"Code snippet FAILED with execution error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "_check_type",
                    "code": code,
                    "result": "failed"
                })

        # Summary output
        print(f"Test Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")  # Expecting only one code snippet

        # ============= Writing test results to test_result.jsonl =============
        # Read existing test_result.jsonl (if exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name == "_check_type"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_check_type"
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