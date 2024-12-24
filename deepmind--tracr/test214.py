import unittest
import json
import os
import numpy as np
from typing import List

TEST_RESULT_JSONL = "test_result.jsonl"

class DummyClass:
    def __init__(self, basis_directions: List[float], magnitudes: np.ndarray):
        self.basis_directions = basis_directions
        self.magnitudes = magnitudes
        self.__post_init__()

    def __post_init__(self):
        """Sort basis directions."""
        if len(self.basis_directions) != self.magnitudes.shape[-1]:
            raise ValueError(
                "Last dimension of magnitudes must be the same as number "
                f"of basis directions. Was {len(self.basis_directions)} "
                f"and {self.magnitudes.shape[-1]}."
            )

        sort_idx = np.argsort(self.basis_directions)
        self.basis_directions = [self.basis_directions[i] for i in sort_idx]
        self.magnitudes = np.take(self.magnitudes, sort_idx, -1)

class TestPostInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[213]  # Get the 214th JSON element

    def test_post_init(self):
        """Test __post_init__ function with various scenarios."""
        results = []
        passed_count = 0
        failed_count = 0

        # Test case 1: Correct sorting
        try:
            basis_directions = [3.0, 1.0, 2.0]
            magnitudes = np.array([1, 2, 3])
            instance = DummyClass(basis_directions, magnitudes)
            self.assertEqual(instance.basis_directions, [1.0, 2.0, 3.0])
            self.assertTrue(np.array_equal(instance.magnitudes, [2, 3, 1]))

            print("Test case 1: PASSED")
            passed_count += 1
            results.append({
                "function_name": "__post_init__",
                "code": self.code_snippet,
                "result": "passed"
            })
        except Exception as e:
            print(f"Test case 1: FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "__post_init__",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Test case 2: Mismatch in length
        try:
            basis_directions = [3.0, 1.0]
            magnitudes = np.array([1, 2, 3])
            with self.assertRaises(ValueError):
                DummyClass(basis_directions, magnitudes)

            print("Test case 2: PASSED")
            passed_count += 1
            results.append({
                "function_name": "__post_init__",
                "code": self.code_snippet,
                "result": "passed"
            })
        except Exception as e:
            print(f"Test case 2: FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "__post_init__",
                "code": self.code_snippet,
                "result": "failed"
            })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 2\n")
        self.assertEqual(passed_count + failed_count, 2, "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__post_init__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()