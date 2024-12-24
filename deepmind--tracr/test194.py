import unittest
import json
import sys
import re
import os
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMakeInputResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[193]  # Get the 194th JSON element

    def test_make_input_snippet(self):
        """Test the make_input function snippet with various checks."""
        passed_count = 0    # Counter for passed tests
        failed_count = 0    # Counter for failed tests
        results = []        # Collect the test results to write to JSONL

        code = self.code_snippet

        print("Running test for make_input code snippet...")

        # ------------------- Static Checks -------------------
        # 1) Static check: See if the function 'make_input' is defined
        if "def make_input" not in code:
            print("Snippet FAILED, function 'make_input' not found.\n")
            failed_count += 1
            # Write the failure record
            results.append({
                "function_name": "make_input",
                "code": code,
                "result": "failed"
            })
            raise AssertionError("Function 'make_input' not found in code.")

        # ------------------- Dynamic Execution & Logic Testing -------------------
        exec_globals = {
            'sys': sys,
            'Any': Any,  # Inject Any
            'one_vec': None,
            'value_vec': None,
            'residual_space': None,
            'bases': None,
        }
        exec_locals = {}

        try:
            # Dynamic execution of the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if the function make_input is actually present
            if 'make_input' not in exec_locals:
                print("Snippet FAILED, 'make_input' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "make_input",
                    "code": code,
                    "result": "failed"
                })
                raise AssertionError("'make_input' not found after executing code.")

            # Mock classes and objects for testing
            class MockBasisDirection:
                def __init__(self, name, x):
                    self.name = name
                    self.x = x

            class MockBases:
                @staticmethod
                def BasisDirection(name, x):
                    return MockBasisDirection(name, x)

            class MockResidualSpace:
                @staticmethod
                def vector_from_basis_direction(basis_direction):
                    return f"residual({basis_direction.x})"

            # Assign mock objects to global variables
            exec_globals['one_vec'] = "1"
            exec_globals['value_vec'] = "v"
            exec_globals['residual_space'] = MockResidualSpace()
            exec_globals['bases'] = MockBases()

            # Test make_input with some sample value
            result = exec_locals['make_input'](5)
            expected_result = "1 + 5 * v + residual(5)"
            self.assertEqual(result, expected_result, "make_input didn't produce expected result.")

            print("Snippet PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "make_input",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Snippet FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "make_input",
                "code": code,
                "result": "failed"
            })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if not present)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "make_input"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "make_input"
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