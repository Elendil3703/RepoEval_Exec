import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAIActionResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[188]  # Get the 189th JSON element (index 188)

    def test_action_function(self):
        """Dynamically test the 'action' function in the code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect the test results to be written to JSONL

        code = self.code_snippet

        func_name = "action"
        if func_name not in code:
            print("Code snippet does not contain 'action' function definition.\n")
            results.append({
                "function_name": func_name,
                "code": code,
                "result": "failed"
            })
        else:
            exec_globals = {
                'Any': Any  # Inject Any if needed
            }
            exec_locals = {}

            try:
                # Dynamically execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Check if 'action' actually exists in exec_locals
                if func_name not in exec_locals:
                    raise ValueError(f"Function '{func_name}' not found in exec_locals.")

                mock_bases = type('bases', (), {
                    'BasisDirection': lambda direction: direction,
                    'VectorInBasis': lambda: 0
                })

                mock_hidden_space = type('hidden_space', (), {
                    'vector_from_basis_direction': lambda x: x,
                    'null_vector': lambda: "null"
                })

                # Mock variables and constants
                one_direction = "one_direction"
                num_vals = 3
                large_number = 1000
                value_thresholds = [0.5, 1.0]

                # Define a mock 'hidden_space' object
                exec_locals.update({
                    'bases': mock_bases,
                    'hidden_space': mock_hidden_space,
                    'one_direction': one_direction,
                    'num_vals': num_vals,
                    'large_number': large_number,
                    'value_thresholds': value_thresholds
                })

                # Test cases
                test_cases = [
                    (one_direction, "expected_output_1"),  # replace with expected output
                    ("other_direction", "expected_output_2")  # replace with expected output
                ]

                for direction, expected in test_cases:
                    result = exec_locals[func_name](direction)
                    # Here you would use the actual expected result, for example through mock inputs
                    self.assertEqual(
                        result, expected, 
                        f"Failed for direction '{direction}'"
                    )

                print(f"Code snippet: PASSED all test cases.\n")
                passed_count += 1
                results.append({
                    "function_name": func_name,
                    "code": code,
                    "result": "passed"
                })

            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": func_name,
                    "code": code,
                    "result": "failed"
                })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")

        # ============= Write the test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if not exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the tested function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != func_name
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