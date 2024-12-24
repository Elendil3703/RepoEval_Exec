import unittest
import json
import os
import numpy as np
from typing import Union

TEST_RESULT_JSONL = "test_result.jsonl"


class MockSearchIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index_data = None

    def add(self, data):
        self.index_data = data

    def search(self, query, k):
        indices = np.argsort(np.linalg.norm(self.index_data - query[:, np.newaxis], axis=2), axis=1)[:, :k]
        distances = np.sort(np.linalg.norm(self.index_data - query[:, np.newaxis], axis=2), axis=1)[:, :k]
        return distances, indices


class TestSearchFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[374]  # Get the 375th code snippet

    def test_nearest_neighbour_search(self):
        exec_globals = {
            "np": np,
            "Union": Union,
            "MockSearchIndex": MockSearchIndex,
        }
        exec_locals = {}

        # Execute the code snippet to get the `run` function
        exec(self.code_snippet, exec_globals, exec_locals)

        # Access the `run` function
        run = exec_locals["run"]

        # Mock instance creation
        class MockInstance:
            def _create_index(self, dimension):
                return MockSearchIndex(dimension)

            def _train(self, index, data):
                pass

            @property
            def search_index(self):
                return MockSearchIndex(2)

        instance = MockInstance()

        # Test data
        n_nearest_neighbours = 3
        query_features = np.array([[1.0, 2.0], [3.0, 4.0]])
        index_features = np.array([[0.5, 2.5], [1.5, 1.5], [3.5, 4.5]])

        # Case 1: index_features is None
        distances, indices = run(instance, n_nearest_neighbours, query_features)
        self.assertEqual(distances.shape, (query_features.shape[0], n_nearest_neighbours))
        self.assertEqual(indices.shape, (query_features.shape[0], n_nearest_neighbours))

        # Case 2: index_features is given
        distances, indices = run(instance, n_nearest_neighbours, query_features, index_features)
        self.assertEqual(distances.shape, (query_features.shape[0], n_nearest_neighbours))
        self.assertEqual(indices.shape, (query_features.shape[0], n_nearest_neighbours))

        # Write results to test_result.jsonl
        results = [{
            "function_name": "run",
            "code": self.code_snippet,
            "result": "passed"
        }]

        # Read existing records
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for "run"
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "run"]

        # Append new results
        existing_records.extend(results)

        # Write to JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()