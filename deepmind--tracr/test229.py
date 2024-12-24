import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestApplyFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and get the specified code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[228]  # Get the 229th JSON element (index 228)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the specified JSON element")

    def test_apply_function(self):
        """Test the apply function from the code snippet with various checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        code = self.code_snippet
        with self.subTest(code=code):
            print("Running test for the apply function...")

            # ------------------- Static checks -------------------
            if "def apply" not in code:
                print("FAILED, function 'apply' not found.\n")
                failed_count += 1
                results.append({
                    "function_name": "apply",
                    "code": code,
                    "result": "failed"
                })
                return

            # ------------------- Dynamic execution and testing -------------------
            exec_globals = {
                'Any': Any,  # Inject Any if needed
                'assert': assert_import,  # Inject special assert handling for checking
            }
            exec_locals = {}

            def relu(x):
                return max(0, x)
            
            exec_globals['relu'] = relu

            try:
                # Dynamically execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Check if apply function is in the locals
                if 'apply' not in exec_locals:
                    print("FAILED, 'apply' function not found in exec_locals.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "apply",
                        "code": code,
                        "result": "failed"
                    })
                    return

                # Simulate inputs and apply the function
                class MockBasis:
                    VectorInBasis = Any
                
                class MockSpaces:
                    input_space = 'test_input_space'
                    output_space = 'test_output_space'
                
                class MockResidual:
                    input_space = 'test_space'
                
                mock_basis = MockBasis()
                
                def project(source, target):
                    def proj_func(x):
                        return x  # Simplified project function
                    return proj_func
                
                def assert_import(cond, message='Assertion failed'):
                    if not cond:
                        raise AssertionError(message)
                
                apply_func = exec_locals['apply']
                # Test apply with mock inputs
                mock_x = mock_basis.VectorInBasis()
                try:
                    apply_func(mock_x)
                    print("PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "apply",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "apply",
                        "code": code,
                        "result": "failed"
                    })
            except Exception as e:
                print(f"FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "apply",
                    "code": code,
                    "result": "failed"
                })

        # Write results to test_result.jsonl
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Read existing test_result.jsonl (if exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function "apply"
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "apply"]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl with updated results
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()