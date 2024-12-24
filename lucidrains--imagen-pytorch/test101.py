import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPrepareFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[100]  # Get the 101st JSON element (index 100)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_prepare_function(self):
        """Dynamically test 'prepare' function in the code snippets."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static Checks -------------------
                # Check if 'prepare' function is actually defined in the code
                if "def prepare" not in code:
                    print(f"Code snippet {i}: FAILED, 'prepare' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "prepare",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Logic Testing -------------------
                exec_globals = {
                    'sys': sys,
                    'Any': Any,  # Inject Any
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Create a class instance if it exists in the executed code
                    class_name = [key for key in exec_locals if isinstance(exec_locals[key], type)][0]
                    MyClass = exec_locals[class_name]
                    instance = MyClass()

                    # Initially, 'prepared' flag should not be set
                    instance.prepared = False

                    # Define a mock method for 'validate_and_set_unet_being_trained'
                    def mock_validate_and_set_unet_being_trained(unet_number):
                        pass

                    # Inject the mock method into the instance
                    instance.validate_and_set_unet_being_trained = mock_validate_and_set_unet_being_trained

                    # Call the 'prepare' method
                    instance.prepare()

                    # After calling 'prepare', the prepared flag should be True
                    self.assertTrue(instance.prepared, f"Code snippet {i} did not set 'prepared' to True.")

                    # Test again for already prepared state. It should raise an AssertionError.
                    with self.assertRaises(AssertionError):
                        instance.prepare()

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "prepare",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "prepare",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results into test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "prepare"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "prepare"
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