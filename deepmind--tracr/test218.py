import unittest
import json
import sys
import os
from typing import Sequence

TEST_RESULT_JSONL = "test_result.jsonl"

def stack(cls, vectors: Sequence["VectorInBasis"], axis: int = 0) -> "VectorInBasis":
    for v in vectors[1:]:
        if v.basis_directions != vectors[0].basis_directions:
            raise TypeError(f"Stacking incompatible bases: {vectors[0]} + {v}")
    return cls(vectors[0].basis_directions, np.stack([v.magnitudes for v in vectors], axis=axis))


class TestStackFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[217]  # Get the 218th JSON element (index 217)

    def test_stack(self):
        """Test the 'stack' function with custom logic."""
        import numpy as np

        # Mock class and vectors for testing
        class VectorInBasis:
            def __init__(self, basis_directions, magnitudes):
                self.basis_directions = basis_directions
                self.magnitudes = magnitudes

        mock_vectors = [
            VectorInBasis([1, 0], np.array([1, 2])),
            VectorInBasis([1, 0], np.array([3, 4])),
            VectorInBasis([1, 0], np.array([5, 6])),
        ]

        # Test successful stacking
        result = stack(VectorInBasis, mock_vectors)
        self.assertTrue(np.array_equal(result.magnitudes, np.array([[1, 2], [3, 4], [5, 6]])), "Correct stacking failed!")

        # Test error on incompatible bases
        incompatible_vectors = [
            VectorInBasis([1, 0], np.array([1, 2])),
            VectorInBasis([0, 1], np.array([3, 4])),  # Different basis
        ]
        with self.assertRaises(TypeError, msg="Incompatible bases did not raise TypeError"):
            stack(VectorInBasis, incompatible_vectors)
        
        # Prepare test result record
        results = [{
            "function_name": "stack",
            "code": self.code_snippet,
            "result": "passed" if self._outcome.success else "failed"
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

        # Remove old stack function records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "stack"
        ]

        # Append new results
        existing_records.extend(results)

        # Write back to JSONL
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()