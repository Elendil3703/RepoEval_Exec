import unittest
import json
import os
import sys
from typing import List
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.figure

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPlotVarCost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and extract the 300th code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[299]

    def test_plot_var_cost_function(self):
        """Test the plot_var_cost function from the provided code snippet."""
        
        exec_globals = {
            'jnp': jnp,
            'plt': plt,
            'matplotlib': matplotlib,
            'List': List,
        }
        exec_locals = {}

        try:
            # Execute the code containing the plot_var_cost function
            exec(self.code_snippet, exec_globals, exec_locals)

            # Ensure the function is defined
            self.assertIn('plot_var_cost', exec_locals, "plot_var_cost function not found in the code snippet.")

            # Retrieve the function
            plot_var_cost = exec_locals['plot_var_cost']

            # Define test input data
            media = jnp.array([[1, 2, 3], [4, 5, 6]])
            costs = jnp.array([10, 20, 30])
            names = ["A", "B", "C"]

            # Conduct the test
            try:
                fig = plot_var_cost(media, costs, names)
                self.assertIsInstance(fig, matplotlib.figure.Figure, "The output is not a matplotlib.figure.Figure.")
                print("Code snippet 299: PASSED all assertions.\n")
                result = {"function_name": "plot_var_cost", "code": self.code_snippet, "result": "passed"}
            except Exception as e:
                print(f"Code snippet 299: FAILED with error: {e}\n")
                result = {"function_name": "plot_var_cost", "code": self.code_snippet, "result": "failed"}

        except Exception as e:
            print(f"Code snippet 299: FAILED with error: {e}\n")
            result = {"function_name": "plot_var_cost", "code": self.code_snippet, "result": "failed"}

        # Write test result to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for plot_var_cost
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "plot_var_cost"]

        # Append the new result
        existing_records.append(result)

        # Write the updated records back
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()