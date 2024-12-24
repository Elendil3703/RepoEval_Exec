import unittest
import json
import os
from typing import Union, Any
import jax.numpy as jnp
from custom_module import preprocessing  # Assuming preprocessing is in a custom module

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGenerateStartingValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[331]  # Get the 332nd JSON element (0-based index)

    def test_generate_starting_values(self):
        """Test the _generate_starting_values function."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet

        # Prepare dynamic execution environment
        exec_globals = {
            'jnp': jnp,
            'preprocessing': preprocessing,
            'Union': Union,
            'Any': Any,
        }
        exec_locals = {}

        try:
            # Dynamically execute the provided code snippet
            exec(code, exec_globals, exec_locals)

            # Ensure the _generate_starting_values function exists
            if '_generate_starting_values' not in exec_locals:
                raise ValueError("Function '_generate_starting_values' not found in executed code.")

            # Retrieve the function
            generate_starting_values = exec_locals['_generate_starting_values']

            # Define test inputs
            n_time_periods = 10
            media = jnp.array([[100, 200], [150, 250], [80, 220]])
            media_scaler = preprocessing.CustomScaler()
            budget = 1000
            prices = jnp.array([5, 10])

            # Call the function with test inputs
            result = generate_starting_values(n_time_periods, media, media_scaler, budget, prices)

            # Assert the output is as expected
            expected_output_shape = (2,)  # should match the number of media channels
            self.assertEqual(result.shape, expected_output_shape, "Result shape incorrect.")
            self.assertTrue(jnp.all(result >= 0), "All channel budgets should be non-negative.")

            print(f"Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "_generate_starting_values",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "_generate_starting_values",
                "code": code,
                "result": "failed"
            })

        # Final summary and writing to file
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Writing results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_generate_starting_values"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()