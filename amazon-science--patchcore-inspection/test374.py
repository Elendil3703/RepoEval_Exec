import unittest
import json
import os
import numpy as np
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthFitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[373]  # Get the 374th JSON element (index 373)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected code in the 374th JSON element")

    def test_fit_function(self):
        """Dynamically test the fit function from the code snippet."""
        code = self.code_snippet
        results = []

        exec_globals = {
            'np': np,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if fit function is defined
            if 'fit' not in exec_locals:
                raise AssertionError("Function 'fit' is not found in the executed code.")

            # Create a mock class with the expected method 'fit'
            class MockClass:
                def __init__(self):
                    self.search_index = None

                def reset_index(self):
                    self.search_index = None

                def _create_index(self, dimension: int):
                    # Mock method to create an index
                    return IndexMock()

                def _train(self, index, features: np.ndarray):
                    # Mock training method
                    pass
            
            class IndexMock:
                def add(self, features: np.ndarray):
                    self.features = features

            # Instantiate the mock class
            instance = MockClass()

            # Retrieve the 'fit' function
            fit_func = exec_locals['fit'].__get__(instance, MockClass)

            # Test with a sample data
            features = np.random.rand(10, 5)  # N=10, D=5
            fit_func(features)

            # Verify the results by checking if features are added to the search index
            self.assertIsNotNone(instance.search_index, "Search index should be created.")
            self.assertTrue(hasattr(instance.search_index, 'features'), "Features not found in search index.")
            np.testing.assert_array_equal(instance.search_index.features, features, "Features added do not match the expected input.")

            results.append({
                "function_name": "fit",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            results.append({
                "function_name": "fit",
                "code": code,
                "result": f"failed with error: {e}"
            })

        # Write the results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "fit"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "fit"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()