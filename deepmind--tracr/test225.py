import unittest
import json
import os
from typing import Any  # We still ensure Any is available

TEST_RESULT_JSONL = "test_result.jsonl"

class TestWovResidualFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the specific JSON file and data for the 225th code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[224]  # Get the 225th snippet; index is 224

    def test_w_ov_residual(self):
        """Test the w_ov_residual function of the given code snippet."""
        passed_count = 0
        failed_count = 0
        results = []

        # Extract the code from the snippet
        code = self.code_snippet
        print("Running test for w_ov_residual function...")
        
        # Check for the existence of the function definition
        if "def w_ov_residual" not in code:
            print("FAILED: 'w_ov_residual' function not found in the code snippet.\n")
            failed_count += 1
            results.append({
                "function_name": "w_ov_residual",
                "code": code,
                "result": "failed"
            })
        else:
            # Run the code in a dynamic environment
            exec_globals = {'Any': Any}
            exec_locals = {}

            try:
                # Execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Verify that 'w_ov_residual' is correctly defined
                if 'w_ov_residual' not in exec_locals:
                    print("FAILED: 'w_ov_residual' function was not executed.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "w_ov_residual",
                        "code": code,
                        "result": "failed"
                    })
                else:
                    # Attempt to instantiate and test the function
                    class MockBases:
                        class VectorInBasis:
                            pass

                    class MockProjection:
                        def __call__(self, vector):
                            return vector

                    class MockWov:
                        def __call__(self, x):
                            return x

                    mock_self = type("MockSelf", (), {
                        "residual_space": MockProjection(),
                        "w_ov": MockWov()
                    })
                    mock_vector = MockBases.VectorInBasis()

                    # Test the function
                    result = exec_locals['w_ov_residual'](mock_self, mock_vector)
                    
                    if isinstance(result, MockBases.VectorInBasis):
                        passed_count += 1
                        results.append({
                            "function_name": "w_ov_residual",
                            "code": code,
                            "result": "passed"
                        })
                        print("PASSED: 'w_ov_residual' executed and returned the correct type.\n")
                    else:
                        failed_count += 1
                        results.append({
                            "function_name": "w_ov_residual",
                            "code": code,
                            "result": "failed"
                        })
                        print("FAILED: 'w_ov_residual' returned an incorrect type.\n")

            except Exception as e:
                print(f"FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "w_ov_residual",
                    "code": code,
                    "result": "failed"
                })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        
        # Ensure the total test count matches expected
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write the results to test_result.jsonl file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))
        
        # Remove old records for this function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "w_ov_residual"
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