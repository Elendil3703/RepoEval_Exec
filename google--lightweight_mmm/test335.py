import unittest
import json
import sys
import os
import pandas as pd
import numpy as np
from typing import Sequence  # Make sure Sequence is available in the injected environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestComputeVariances(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        # Get the 335th element (334 index due to zero-based indexing)
        cls.code_snippet = data[334]  # Get the 335th JSON element
        if not cls.code_snippet:
            raise ValueError("Expected the code snippet to be non-empty")
        
        # Placeholder function for core_utils.get_number_geos
        global core_utils
        class CoreUtilsMock:
            @staticmethod
            def get_number_geos(features):
                return features.shape[-1] if features.ndim > 1 else 1
        core_utils = CoreUtilsMock()

    def test_compute_variances(self):
        """Test _compute_variances function equivalence."""
        passed_count = 0
        failed_count = 0
        results = []

        exec_globals = {
            'jnp': np,
            'pd': pd,
            'np': np,
            'core_utils': core_utils,
            'copy': __import__('copy'),
            'Sequence': Sequence
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Ensure the function is present
            if '_compute_variances' not in exec_locals:
                print(f"Code snippet: FAILED, '_compute_variances' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "_compute_variances",
                    "code": self.code_snippet,
                    "result": "failed"
                })
                return

            # Sample test case
            features = np.array([
                [0.1, 0.3, 0.5],
                [0.2, 0.4, 0.6]
            ])
            feature_names = ["feature1", "feature2"]
            geo_names = ["geo1", "geo2", "geo3"]

            try:
                result_df = exec_locals['_compute_variances'](features, feature_names, geo_names)

                # Check the dataframe structure
                self.assertEqual(result_df.shape, (2, 3), "Unexpected dataframe shape")
                self.assertListEqual(list(result_df.columns), geo_names, "Unexpected column names")
                self.assertListEqual(list(result_df.index), feature_names, "Unexpected index names")

                print("Code snippet: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "_compute_variances",
                    "code": self.code_snippet,
                    "result": "passed"
                })
            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "_compute_variances",
                    "code": self.code_snippet,
                    "result": "failed"
                })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "_compute_variances",
                "code": self.code_snippet,
                "result": "failed"
            })
        
        # Test Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")

        # Ensure test count matches expectations
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "_compute_variances"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_compute_variances"
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