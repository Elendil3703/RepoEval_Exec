import unittest
import json
import os
from typing import List, Any

TEST_RESULT_JSONL = "test_result.jsonl"


class VectorInBasis:
    def __init__(self, basis_directions: List[float], magnitudes: List[float]):
        self.basis_directions = basis_directions
        self.magnitudes = magnitudes

    def __add__(self, other: "VectorInBasis") -> "VectorInBasis":
        if self.basis_directions != other.basis_directions:
            raise TypeError(f"Adding incompatible bases: {self} + {other}")
        magnitudes = self.magnitudes + other.magnitudes
        return VectorInBasis(self.basis_directions, magnitudes)


class TestVectorInBasisAdd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[214]  # Get the 215th JSON element (index 214)

    def test_vector_addition(self):
        """Test the __add__ method of VectorInBasis class."""
        results = []  # Collect results to write to JSONL

        # ------------------- Testing logic -------------------
        try:
            # Prepare data for testing
            vector1 = VectorInBasis([1, 0], [2, 3])
            vector2 = VectorInBasis([1, 0], [4, 5])
            vector3 = VectorInBasis([0, 1], [7, 8])  # Different basis directions

            # Test addition with the same basis directions
            result_vector = vector1 + vector2
            expected_magnitudes = [6, 8]
            self.assertEqual(
                result_vector.magnitudes,
                expected_magnitudes,
                "Magnitudes do not match the expected values."
            )
            results.append({
                "function_name": "__add__",
                "code": self.code_snippet,
                "result": "passed"
            })
            print("Vector addition with same basis directions: PASSED.")

            # Test addition with different basis directions
            with self.assertRaises(TypeError):
                _ = vector1 + vector3
            results.append({
                "function_name": "__add__",
                "code": self.code_snippet,
                "result": "passed"
            })
            print("Vector addition with different basis directions: TypeError raised as expected.")

        except Exception as e:
            print(f"Test failed with error: {e}")
            results.append({
                "function_name": "__add__",
                "code": self.code_snippet,
                "result": "failed"
            })

        # ============= Update test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for __add__
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__add__"
        ]

        # Add new results
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()