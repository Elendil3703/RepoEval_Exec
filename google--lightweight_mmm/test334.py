import unittest
import json
import os
import sys
import jax.numpy as jnp
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class NotFittedScalerError(Exception):
    """Custom exception to be raised if the scaler is used before fitting."""
    pass

class TestTransformFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[333]  # Get the 334th JSON element (334th is index 333)

    def test_transform_function(self):
        """Test the transform function in the provided code snippet."""
        code = self.code_snippet

        results = []  # Collect results to be written to JSONL

        try:
            exec_globals = {
                'jnp': jnp,
                'NotFittedScalerError': NotFittedScalerError,
                '__name__': '__main__'
            }
            exec_locals = {}

            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if 'transform' is a method of a class in exec_locals
            transform_found = False
            for obj in exec_locals.values():
                if hasattr(obj, 'transform'):
                    # Assume that transform is a method of a class
                    transform = getattr(obj, 'transform')
                    transform_found = True
                    break

            # Assert that we have discovered a transform method
            self.assertTrue(transform_found, "No 'transform' method found in the provided class.")

            # Create an instance of the class (assuming it doesn't require init arguments)
            scaler_instance = obj()

            # Test transform behavior without fitting
            with self.assertRaises(NotFittedScalerError):
                scaler_instance.transform(jnp.array([1, 2, 3]))

            # Manually set attributes to simulate fitting
            scaler_instance.divide_by = 2.0
            scaler_instance.multiply_by = 4.0

            # Test correct transformation
            input_data = jnp.array([1, 2, 3])
            expected_output = jnp.array([2, 4, 6])
            output_data = scaler_instance.transform(input_data)
            jnp.testing.assert_array_equal(output_data, expected_output)

            print("Code snippet: PASSED all assertions.\n")
            results.append({
                "function_name": "transform",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            results.append({
                "function_name": "transform",
                "code": code,
                "result": "failed"
            })

        # ============= Write test results to test_result.jsonl =============
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
            if rec.get("function_name") != "transform"
        ]

        # Add new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()