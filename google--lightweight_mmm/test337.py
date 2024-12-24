import unittest
import json
import os
import jax.numpy as jnp
import pandas as pd
from typing import Sequence
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import copy

TEST_RESULT_JSONL = "test_result.jsonl"

class TestComputeVarianceInflationFactors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[336]  # Get the 337th code snippet

        if not cls.code_snippet.strip():
            raise ValueError("Code snippet is empty.")

        # Execute the code snippet to define the function
        exec_globals = {
            'jnp': jnp,
            'pd': pd,
            'Sequence': Sequence,
            'variance_inflation_factor': variance_inflation_factor,
            'add_constant': add_constant,
            'copy': copy,
            'core_utils': type('core_utils', (), {'get_number_geos': lambda x: x.shape[-1] if x.ndim > 1 else 1}),
        }
        exec(cls.code_snippet, exec_globals)
        cls._compute_variance_inflation_factors = exec_globals['_compute_variance_inflation_factors']

    def test_vif_computation(self):
        """Test the _compute_variance_inflation_factors function with specific data."""
        # Sample input setup
        features = jnp.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0]
        ])
        feature_names = ["feature1", "feature2", "feature3"]
        geo_names = ["geo1"]

        try:
            # Call the VIF computation function
            vif_df = self._compute_variance_inflation_factors(features, feature_names, geo_names)
            # Check that the returned dataframe has the expected shape
            self.assertEqual(vif_df.shape, (3, 1))
            # Check that the dataframe has the correct column name
            self.assertEqual(list(vif_df.columns), geo_names)
            # Check that the dataframe has the correct index (feature names)
            self.assertEqual(list(vif_df.index), feature_names)
            # Further checks can be added to validate individual VIF values, if needed
            test_result = "passed"
        except ValueError as e:
            if "does not match the length of geo_names" in str(e):
                test_result = "failed"
            else:
                raise e

        results = [{
            "function_name": "_compute_variance_inflation_factors",
            "code": self.code_snippet,
            "result": test_result
        }]

        # Log the results in a JSONL file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                existing_records = [json.loads(line) for line in f]

        # Filter out old results for this function
        existing_records = [
            record for record in existing_records
            if record.get("function_name") != "_compute_variance_inflation_factors"
        ]

        # Append the new result
        existing_records.extend(results)

        # Write back the updated results to the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()