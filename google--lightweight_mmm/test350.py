import unittest
import json
import os
import numpy as np
import jax
import jax.numpy as jnp

TEST_RESULT_JSONL = "test_result.jsonl"

class TestAdstockFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[349]  # Get the 350th JSON element

    def test_adstock_function(self):
        """Dynamically test the adstock function from the provided code snippet."""
        passed = False
        results = []

        code = self.code_snippet
        exec_globals = {
            'jax': jax,
            'jnp': jnp,
            'np': np,
        }
        exec_locals = {}

        try:
            # Dynamic execution of the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if adstock is defined
            if 'adstock' not in exec_locals:
                print("Adstock function not found in executed code.")
                results.append({
                    "function_name": "adstock",
                    "code": code,
                    "result": "failed"
                })
            else:
                adstock_fn = exec_locals['adstock']

                # Test cases for the adstock function
                test_data = jnp.array([1.0, 0.5, 0.0, 1.0, 0.5])
                expected_normalised = np.array([0.1, 0.59, 0.531, 1.4779, 1.83011])
                expected_unnormalised = np.array([1.0, 1.4, 1.26, 3.134, 3.8206])

                # Normalize case
                result_normalised = adstock_fn(test_data, lag_weight=0.9, normalise=True)
                np.testing.assert_almost_equal(result_normalised, expected_normalised, decimal=5)

                # Unnormalized case
                result_unnormalised = adstock_fn(test_data, lag_weight=0.9, normalise=False)
                np.testing.assert_almost_equal(result_unnormalised, expected_unnormalised, decimal=5)

                passed = True
                results.append({
                    "function_name": "adstock",
                    "code": code,
                    "result": "passed"
                })
        except Exception as e:
            print(f"Adstock test failed with the error: {e}")
            results.append({
                "function_name": "adstock",
                "code": code,
                "result": "failed"
            })

        # ============= Write test results to test_result.jsonl =============
        # Read existing records from test_result.jsonl (ignore if non-existent)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "adstock"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "adstock"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")
        if passed:
            print("Adstock function passed all tests.")

if __name__ == "__main__":
    unittest.main()