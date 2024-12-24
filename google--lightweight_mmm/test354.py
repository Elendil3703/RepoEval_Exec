import unittest
import json
import os
from typing import Any, MutableMapping, Sequence
import numpy as np
import jax.numpy as jnp

TEST_RESULT_JSONL = "test_result.jsonl"

def _compare_equality_for_lmmm(item_1: Any, item_2: Any) -> bool:
    """Compares two items for equality.

    Helper function for the __eq__ method of LightweightmMM. First checks if items
    are strings or lists of strings (it's okay if empty lists compare True), then
    uses jnp.array_equal if the items are jax.numpy.DeviceArray or other related
    sequences, and uses items' __eq__ otherwise.

    Note: this implementation does not cover every possible data structure, but
    it does cover all the data structures seen in attributes used by
    LightweightMMM. Sometimes the DeviceArray is hidden in the value of a
    MutableMapping, hence the recursion.

    Args:
      item_1: First item to be compared.
      item_2: Second item to be compared.

    Returns:
      Boolean for whether item_1 equals item_2.
    """
    if type(item_1) != type(item_2):
        is_equal = False
    elif isinstance(item_1, str):
        is_equal = item_1 == item_2
    elif isinstance(item_1, (jnp.ndarray, np.ndarray, Sequence)):
        if all(isinstance(x, str) for x in item_1) and all(isinstance(x, str) for x in item_2):
            is_equal = item_1 == item_2
        else:
            is_equal = np.array_equal(item_1, item_2, equal_nan=True)
    elif isinstance(item_1, MutableMapping):
        is_equal = all([
            _compare_equality_for_lmmm(item_1[x], item_2[x])
            for x in item_1.keys() | item_2.keys()
        ])
    else:
        is_equal = item_1 == item_2

    return is_equal

class TestCompareEqualityForLMMM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[353]  # Get the 354th JSON element

    def test_compare_equality_for_lmmm(self):
        """Test the _compare_equality_for_lmmm function with various cases."""
        passed_count = 0
        failed_count = 0
        results = []

        test_cases = [
            {"input": ("hello", "hello"), "expected": True},
            {"input": (jnp.array([1, 2, 3]), jnp.array([1, 2, 3])), "expected": True},
            {"input": (np.array([1, np.nan]), np.array([1, np.nan])), "expected": True},
            {"input": ({"a": 1}, {"a": 1}), "expected": True},
            {"input": ("hello", "world"), "expected": False},
            {"input": (jnp.array([1, 2, 3]), jnp.array([1, 2, 4])), "expected": False},
            {"input": (np.array([1, np.nan]), np.array([1, 2])), "expected": False},
            {"input": ({"a": 1}, {"b": 1}), "expected": False},
        ]

        for i, case in enumerate(test_cases):
            with self.subTest(case_index=i):
                input_data = case['input']
                expected = case['expected']
                try:
                    result = _compare_equality_for_lmmm(*input_data)
                    self.assertEqual(result, expected)
                    passed_count += 1
                    test_result = "passed"
                except AssertionError as e:
                    failed_count += 1
                    test_result = "failed"

                results.append({
                    "function_name": "_compare_equality_for_lmmm",
                    "input": input_data,
                    "expected": expected,
                    "result": test_result
                })

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with the same function_name
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_compare_equality_for_lmmm"
        ]

        # Extend with new results
        existing_records.extend(results)

        # Write back to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()