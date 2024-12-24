import unittest
import json
import os
from typing import Any, Tuple
import numpy as np

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and extract the 83rd element (index 82)
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[82]  # Get the 83rd JSON element
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet at index 82")

    def test_forward_function(self):
        """Dynamically test the forward function in the provided code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to the JSONL file

        code = self.code_snippet
        with self.subTest():
            print(f"Running test for forward function...")

            # Check if forward function is present in the code
            if "def forward" not in code:
                print("Code snippet: FAILED, function 'forward' not found.\n")
                failed_count += 1
                results.append({
                    "function_name": "forward",
                    "code": code,
                    "result": "failed"
                })
                return

            # Prepare exec environment
            exec_locals = {
                'rearrange': rearrange,
                'pack': pack,
                'unpack': unpack,
                'np': np,
                'Any': Any,
            }

            try:
                # Execute code to define the forward function
                exec(code, globals(), exec_locals)

                # Ensure forward function exists after execution
                if 'forward' not in exec_locals:
                    print("Code snippet: FAILED, 'forward' not found in exec_locals.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })
                    return

                forward = exec_locals['forward']

                # Define test inputs and expected behavior
                class MockLayer:
                    def __init__(self, kind: str):
                        self.kind = kind

                    def __call__(self, x, context=None):
                        return x + 1

                class MockSelf:
                    def __init__(self):
                        self.layers = [(MockLayer('attn'), MockLayer('ff'))]

                mock_self = MockSelf()
                input_array = np.ones((2, 3, 4, 5))  # Example input

                # Call forward function
                result = forward(mock_self, input_array)

                # Validate the result
                expected_output_shape = input_array.shape
                self.assertEqual(result.shape, expected_output_shape, f"Expected output shape {expected_output_shape}, but got {result.shape}.")
                
                print("Code snippet: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "forward",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "forward",
                    "code": code,
                    "result": "failed"
                })

        # Summary and writing results to file
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for forward function
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "forward"]

        # Add new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

# Mock functions to simulate rearrange, pack, and unpack
def rearrange(x, pattern):
    return x

def pack(lst, pattern):
    return lst[0], (1, 2, 3)  # Mock return values

def unpack(x, ps, pattern):
    return (x,)

if __name__ == "__main__":
    unittest.main()