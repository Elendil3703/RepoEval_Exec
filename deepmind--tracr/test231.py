import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestApplyFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[230]  # Get the 231st JSON element
        if not cls.code_snippet:
            raise ValueError("Expected code snippet at index 230 in the JSON data")

    def test_apply_function(self):
        """Test the 'apply' function from code snippet 230."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        print("Testing code snippet for 'apply' function...")
        
        if "def apply" not in code:
            print("Code snippet: FAILED, 'apply' function not defined.\n")
            failed_count += 1
            results.append({
                "function_name": "apply",
                "code": code,
                "result": "failed"
            })
        else:
            try:
                # Create a mock environment for testing
                exec_globals = {
                    'bases': __import__('types', fromlist=['SimpleNamespace']),
                    'Any': Any,
                }
                exec_locals = {}

                # Define a mock VectorInBasis and Block to simulate the test
                exec_globals['bases'].VectorInBasis = exec_globals['bases'].SimpleNamespace
                MockBlock = exec_globals['bases'].SimpleNamespace
                MockBlock.residual_space = None
                MockBlock.apply = lambda self, x: x

                # Prepare the testing environment
                class Mock:
                    residual_space = None
                    blocks = [MockBlock()]  # Single block for simplicity

                # Execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Instantiate the object
                obj = Mock()

                # Replace real code with mocks
                exec_locals['self'] = obj
                exec_locals['x'] = exec_globals['bases'].VectorInBasis()

                # Test the 'apply' function
                result = exec_locals['apply'](obj, exec_locals['x'])

                # Assert that the function executes and returns a value
                self.assertIsNotNone(result, "apply function did not return a result.")
                
                print("Code snippet: PASSED.\n")
                passed_count += 1
                results.append({
                    "function_name": "apply",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "apply",
                    "code": code,
                    "result": "failed"
                })

        # Final summary
        print(f"Test Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for 'apply'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "apply"
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