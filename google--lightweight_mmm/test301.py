import unittest
import json
import sys
import re
import os
from typing import Any
import jax.numpy as jnp
import matplotlib.pyplot as plt

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCallFitPlotter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[300]  # Get the 301st JSON element (0-based index)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the 301st JSON array")

    def test_call_fit_plotter(self):
        """Dynamically test the _call_fit_plotter function in the JSON with additional checks."""
        results = []  # Collect test results to write to JSONL

        code = self.code_snippet

        # ------------ Extra logic checks ----------------
        # 1) Static check: Ensure "_call_fit_plotter" is defined in the snippet
        if "_call_fit_plotter" not in code:
            print(f"Code snippet does not contain '_call_fit_plotter'.")
            results.append({
                "function_name": "_call_fit_plotter",
                "code": code,
                "result": "failed"
            })
            return

        # 2) Ensure function definition follows expected form
        func_pattern = r"def\s+_call_fit_plotter\s*\("
        if not re.search(func_pattern, code):
            print(f"Code snippet has incorrect signature for '_call_fit_plotter'.")
            results.append({
                "function_name": "_call_fit_plotter",
                "code": code,
                "result": "failed"
            })
            return

        # ------------ Dynamic execution and testing logic ----------------
        exec_globals = {
            'jnp': jnp,
            'plt': plt
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check _call_fit_plotter exists in executed locals
            self.assertIn('_call_fit_plotter', exec_locals)

            # Retrieve the function
            _call_fit_plotter = exec_locals['_call_fit_plotter']

            # Define test data
            predictions_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
            target_2d = jnp.array([1.5, 3.5])
            interval_mid_range = 0.9
            digits = 2

            # Execute function and test result
            result_figure = _call_fit_plotter(predictions_2d, target_2d, interval_mid_range, digits)
            self.assertIsInstance(result_figure, plt.Figure, "The output is not a matplotlib.figure.Figure instance.")

            # Add more checks for additional scenarios
            predictions_3d = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            target_3d = jnp.array([[1.5, 3.5], [5.5, 7.5]])
            result_figure = _call_fit_plotter(predictions_3d, target_3d, interval_mid_range, digits)
            self.assertIsInstance(result_figure, plt.Figure, "The output is not a matplotlib.figure.Figure instance.")

            print(f"Code snippet: PASSED all assertions.\n")
            results.append({
                "function_name": "_call_fit_plotter",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Code snippet failed with error: {e}\n")
            results.append({
                "function_name": "_call_fit_plotter",
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

        # Remove old records for _call_fit_plotter
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_call_fit_plotter"
        ]

        # Append the new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()