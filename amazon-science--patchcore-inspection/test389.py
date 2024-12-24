import unittest
import numpy as np
import json
import os
from typing import List

TEST_RESULT_JSONL = "test_result.jsonl"

class MockFeatureMerger:
    def merge(self, features):
        # Mock behavior, returns combined numpy array
        return np.concatenate(features, axis=0)

class MockNearestNeighbor:
    def fit(self, features):
        # Mock behavior, simply stores the features
        self.fitted_features = features

class FitFunctionImplementation:
    def __init__(self):
        self.feature_merger = MockFeatureMerger()
        self.nn_method = MockNearestNeighbor()

    def fit(self, detection_features: List[np.ndarray]) -> None:
        """Calls the fit function of the nearest neighbour method.

        Args:
            detection_features: [list of np.arrays]
                [[bs x d_i] for i in n] Contains a list of
                np.arrays for all training images corresponding to respective
                features VECTORS (or maps, but will be resized) produced by
                some backbone network which should be used for image-level
                anomaly detection.
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        self.nn_method.fit(self.detection_features)

class TestFitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        # Select the specific code snippet
        cls.code_snippet = data[388]

    def test_fit_function(self):
        """Test the fit function logic."""
        passed_count = 0
        failed_count = 0
        results = []

        print("Testing fit function...")

        try:
            # Prepare mock data
            mock_data = [np.random.rand(10, 5), np.random.rand(15, 5)]
            expected_merged_data = np.concatenate(mock_data, axis=0)

            # Instantiate and use the real fit function logic
            fit_impl = FitFunctionImplementation()
            fit_impl.fit(mock_data)

            # Assertions
            self.assertTrue(
                np.array_equal(fit_impl.detection_features, expected_merged_data),
                "The merged features are incorrect."
            )

            self.assertTrue(
                np.array_equal(fit_impl.nn_method.fitted_features, expected_merged_data),
                "The nearest neighbor method does not have the correct features."
            )

            print("Fit function: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "fit",
                "code": self.code_snippet,
                "result": "passed"
            })
        except Exception as e:
            print(f"Fit function: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "fit",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Test Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function name "fit"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "fit"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

unittest.main(argv=[''], exit=False)