import unittest
import json
import sys
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestOperationFnResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_data = data[197]  # Get the 198th JSON element
        if len(cls.code_data) < 1:
            raise ValueError("Expected at least one code snippet in the 198th JSON array")

    def test_operation_fn_logic(self):
        """Dynamically test operation_fn logic in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_data):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Dynamic Code Execution and Logic Testing -------------------
                exec_globals = {
                    'input_space': {"NORTH", "SOUTH", "EAST", "WEST"},  # mocking input_space
                    'output_space': self.MockOutputSpace(),  # mocking output_space
                    'operation': self.mock_operation,  # mocking operation function
                    'out_vec': 10,  # example out_vec
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if operation_fn is present
                    if 'operation_fn' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'operation_fn' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "operation_fn",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Testing the logic with mock direction
                    operation_fn = exec_locals['operation_fn']

                    # Valid directions test
                    for direction in exec_globals['input_space']:
                        self.assertEqual(
                            operation_fn(direction),
                            self.mock_operation(direction) * exec_globals['out_vec'],
                            f"Code snippet {i} failed for direction {direction}."
                        )

                    # Invalid direction test
                    self.assertEqual(
                        operation_fn("INVALID"),
                        exec_globals['output_space'].null_vector(),
                        f"Code snippet {i} failed for invalid direction."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "operation_fn",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "operation_fn",
                        "code": code,
                        "result": "failed"
                    })

        # Summary of test results
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_data)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_data), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if it doesn't exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name == "operation_fn"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "operation_fn"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

    class MockOutputSpace:
        def null_vector(self):
            return 0  # mocked null vector

    @staticmethod
    def mock_operation(direction):
        return 1  # mocked operation

if __name__ == "__main__":
    unittest.main()