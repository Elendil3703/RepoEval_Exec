import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestNumStepsTaken(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[105]  # Get the 106th (index 105) JSON element
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet at index 105")

    def test_num_steps_taken(self):
        """Test the num_steps_taken function with various cases."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect and write test results to JSONL

        for i, code in enumerate([self.code_snippet]):  # Wrap in list for loop structure
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Create an execution environment with necessary imports and variables
                exec_globals = {}
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Assuming the function is within a class, instantiate and test it
                    test_class = type('TestClass', (object,), exec_locals)
                    instance = test_class()
                    instance.num_unets = 1
                    instance.steps = [3]  # Assuming a list/array for steps

                    # Test cases
                    result = instance.num_steps_taken()  # Should use default for unet_number
                    self.assertEqual(result, 3, f"Code snippet {i} failed on single unet.")

                    instance.num_unets = 2
                    instance.steps = [3, 5]
                    result = instance.num_steps_taken(unet_number=2)
                    self.assertEqual(result, 5, f"Code snippet {i} failed on unet_number=2.")

                    result = instance.num_steps_taken(unet_number=1)
                    self.assertEqual(result, 3, f"Code snippet {i} failed on unet_number=1.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "num_steps_taken",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "num_steps_taken",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {1}\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write the results to test_result.jsonl
        # Read existing test_result.jsonl (ignore if doesn't exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function_name "num_steps_taken"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "num_steps_taken"
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