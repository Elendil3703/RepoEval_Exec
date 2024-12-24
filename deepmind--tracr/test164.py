import unittest
import json
import os
from typing import Any, Callable, Optional  # Ensure Callable and Optional are available

TEST_RESULT_JSONL = "test_result.jsonl"

class BasisDirection:
    """Mock class for BasisDirection used in tests."""
    def __init__(self, direction_name, value):
        self.direction_name = direction_name
        self.value = value

    def __eq__(self, other):
        return (self.direction_name == other.direction_name and
                self.value == other.value)

class TestTransformFunToBasisFun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load RepoEval_result.json and get the 164th snippet (index 163)
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[163]  # Get the 164th JSON element
        
        # Inject mock 'bases' module
        cls.mock_globals = {'bases': {'BasisDirection': BasisDirection}}
        
    def test_transform_fun_to_basis_fun(self):
        """Test _transform_fun_to_basis_fun transformation logic on various functions."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results for writing to JSONL
        
        # Define a sample function to transform
        def sample_fun(x, y):
            return x + y

        # Globals for exec, including mocked 'bases'
        exec_globals = {'Any': Any, 'Optional': Optional, 'Callable': Callable, **self.mock_globals}
        exec_locals = {}

        try:
            # Inject and execute the provided code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check if _transform_fun_to_basis_fun is defined
            if '_transform_fun_to_basis_fun' not in exec_locals:
                print("Code snippet: FAILED, '_transform_fun_to_basis_fun' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "_transform_fun_to_basis_fun",
                    "code": self.code_snippet,
                    "result": "failed"
                })
                return
            
            # Retrieve the transformation function
            transform_fun = exec_locals['_transform_fun_to_basis_fun']

            # Test case 1: Without direction name
            transformed_fun = transform_fun(sample_fun)
            result = transformed_fun(BasisDirection("", 1), BasisDirection("", 2))
            expected = 3
            self.assertEqual(result, expected, "Test case 1 failed")
            
            # Test case 2: With direction name
            transformed_fun_named = transform_fun(sample_fun, "output_direction")
            result_named = transformed_fun_named(BasisDirection("", 1), BasisDirection("", 2))
            expected_named = BasisDirection("output_direction", 3)
            self.assertEqual(result_named, expected_named, "Test case 2 failed")

            print("Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "_transform_fun_to_basis_fun",
                "code": self.code_snippet,
                "result": "passed"
            })
            
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "_transform_fun_to_basis_fun",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")  # As we test only one code snippet

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for _transform_fun_to_basis_fun
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_transform_fun_to_basis_fun"
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