import unittest
import json
import sys
import os
from typing import List, Union
import numpy as np

TEST_RESULT_JSONL = "test_result.jsonl"

class MockFeatureMerger:
    def merge(self, query_features):
        # Mock behavior for feature merging
        return query_features

class MockNearestNeighbor:
    def __call__(self, query_features):
        # Mock behavior for the nearest neighbor calculation
        query_distances = np.random.rand(len(query_features), 5)
        query_nns = np.random.randint(0, 100, (len(query_features), 5))
        return query_distances, query_nns

class TestPredictFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[389]  # Get the 390th JSON element

    def test_predict_function(self):
        """Test the predict function from the code snippet."""
        results = []  # Collect results for writing to JSONL

        try:
            namespace = {
                'np': np,
                'List': List,
                'Union': Union,
            }

            exec(self.code_snippet, namespace)

            # Check if 'predict' is defined
            if 'predict' not in namespace:
                raise AssertionError("'predict' not found in the code snippet.")

            # Creating mock objects and dummy query_features
            mock_feature_merger = MockFeatureMerger()
            mock_imagelevel_nn = MockNearestNeighbor()
            query_features = [np.random.rand(128) for _ in range(10)]

            # Execute the predict function
            predict_func = namespace['predict']
            predict_func = predict_func.__get__(self)

            # Inject mocks into the instance
            self.feature_merger = mock_feature_merger
            self.imagelevel_nn = mock_imagelevel_nn

            anomaly_scores, query_distances, query_nns = predict_func(query_features)

            # Assertions to check predict output format
            self.assertIsInstance(anomaly_scores, np.ndarray, "anomaly_scores should be a numpy array")
            self.assertEqual(anomaly_scores.shape[0], len(query_features), "anomaly_scores length mismatch")
            self.assertIsInstance(query_distances, np.ndarray, "query_distances should be a numpy array")
            self.assertIsInstance(query_nns, np.ndarray, "query_nns should be a numpy array")

            print(f"Code snippet 'predict': PASSED all assertions.\n")
            results.append({
                "function_name": "predict",
                "code": self.code_snippet,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet 'predict': FAILED with error: {e}\n")
            results.append({
                "function_name": "predict",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Reading existing test results
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for 'predict'
        existing_records = [
            rec for rec in existing_records if rec.get("function_name") != "predict"
        ]

        # Append new results
        existing_records.extend(results)

        # Write back to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()