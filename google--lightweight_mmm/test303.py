import unittest
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"


class TestPlotOutOfSampleModelFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[302]
        if not cls.code_snippet:
            raise ValueError("Expected code snippet not found.")

    def test_plot_out_of_sample_model_fit(self):
        """Test the function plot_out_of_sample_model_fit."""
        exec_globals = {
            'jnp': np,  # Using numpy as a stand-in for jnp
            'matplotlib': plt,
            '_call_fit_plotter': self.mocked_call_fit_plotter  # Mocking the function
        }
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check if function exists in exec_locals
            func_name = 'plot_out_of_sample_model_fit'
            if func_name not in exec_locals:
                raise AssertionError(f"Function '{func_name}' not found in executed code.")

            # Get the function to test
            plot_out_of_sample_model_fit = exec_locals[func_name]

            # Test data
            predictions = np.array([1.0, 2.0, 3.0])
            target = np.array([1.5, 2.5, 3.5])
            
            # Call the function
            result = plot_out_of_sample_model_fit(predictions, target)

            # Check if the result is a matplotlib figure
            self.assertIsInstance(result, plt.Figure, "The result should be a matplotlib.figure.Figure.")

            print("Test passed: plot_out_of_sample_model_fit.")

            # Save results
            results = [{
                "function_name": func_name,
                "code": self.code_snippet,
                "result": "passed"
            }]
        except Exception as e:
            print(f"Test failed with error: {e}")
            # Save results
            results = [{
                "function_name": func_name,
                "code": self.code_snippet,
                "result": "failed"
            }]

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the same function
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

    def mocked_call_fit_plotter(self, predictions, target, interval_mid_range, digits):
        """Mocked _call_fit_plotter for testing purposes."""
        fig, ax = plt.subplots()
        ax.plot(predictions, label='Predictions')
        ax.plot(target, label='Target')
        ax.legend()
        return fig


if __name__ == "__main__":
    unittest.main()