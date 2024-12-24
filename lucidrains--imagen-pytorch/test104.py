import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSetAcceleratorScaler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[103]  # Get the 104th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 104th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Static check: Look for 'set_accelerator_scaler' in the snippet
                if "def set_accelerator_scaler" not in code:
                    print(f"Code snippet {i}: FAILED, 'set_accelerator_scaler' not found in code.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "set_accelerator_scaler",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and logical testing
                exec_globals = {
                    'Any': Any,  # Injection of Any
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Define a mock class to test set_accelerator_scaler function
                    class MockAccelerator:
                        def __init__(self):
                            self._optimizers = [MockOptimizer() for _ in range(3)]

                    class MockOptimizer:
                        def __init__(self):
                            self.scaler = None

                    class TestClass:
                        def __init__(self):
                            self.accelerator = MockAccelerator()
                            self.scaler0 = "scaler0"
                            self.scaler1 = "scaler1"
                            self.scaler2 = "scaler2"

                        def validate_unet_number(self, unet_number):
                            return unet_number  # Assume validation passes

                    # Instantiate the test class and set up the environment
                    instance = TestClass()
                    unet_number = 2  # Example unet_number
                    exec_locals['set_accelerator_scaler'](instance, unet_number)

                    # Verification checks
                    expected_scaler = instance.scaler1
                    self.assertEqual(instance.accelerator.scaler, expected_scaler,
                                     f"Code snippet {i} did not correctly set the accelerator scaler.")

                    for optimizer in instance.accelerator._optimizers:
                        self.assertEqual(optimizer.scaler, expected_scaler,
                                         f"Code snippet {i} did not correctly set the optimizer scaler.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "set_accelerator_scaler",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "set_accelerator_scaler",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "set_accelerator_scaler"
        ]

        # Append the new results
        existing_records.extend(results)

        # Re-write test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()