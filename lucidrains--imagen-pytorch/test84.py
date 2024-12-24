import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthInit(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load JSON data
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)

        # Load the specific code snippet by index
        cls.code_snippet = data[83] 
        if not cls.code_snippet:
            raise ValueError("Expected code snippet at index 83.")

    def test_init_function(self):
        """Test for the __init__ function of a neural network module."""
        passed_count = 0
        failed_count = 0
        results = []
        
        # Functions to be tested (assuming they're defined in the code snippet)
        code = self.code_snippet

        # Prepare for execution
        exec_globals = {
            'nn': __import__('torch').nn,
            'default': lambda x, y: x if x is not None else y
        }
        exec_locals = {}

        try:
            # Execute the code snippet dynamically
            exec(code, exec_globals, exec_locals)

            # Check if the __init__ method is defined in a class
            found_init = False
            for obj_name, obj in exec_locals.items():
                if isinstance(obj, type) and '__init__' in obj.__dict__:
                    init_func = obj.__dict__['__init__']
                    found_init = True

                    # Run tests on __init__
                    self._test_assert_conditions(init_func)  # You can replace with actual tests

                    print("Test for __init__ function: PASSED.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })

            if not found_init:
                print("FAILED: __init__ function not found in any class.\n")
                results.append({
                    "function_name": "__init__",
                    "code": code,
                    "result": "failed"
                })
                failed_count += 1

        except Exception as e:
            print(f"FAILED with error: {e}\n")
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })
            failed_count += 1

        # Summary
        print(f"Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch")

        # Write results to JSONL
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_records.append(json.loads(line))

        existing_records = [rec for rec in existing_records if rec.get("function_name") != "__init__"]
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

    def _test_assert_conditions(self, init_func):
        # Example tests on the __init__ method
        # Replace with relevant logic based on your needs
        
        # Sample parameter checks
        try:
            # Assert kernel sizes and stride logic
            init_func(self=None, dim_in=16, kernel_sizes=[3, 5, 7], dim_out=32, stride=1)
            init_func(self=None, dim_in=16, kernel_sizes=[3, 5, 7], dim_out=32, stride=2)
        except AssertionError:
            self.fail("Assertion in __init__ failed for valid inputs")


if __name__ == "__main__":
    unittest.main()