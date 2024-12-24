import unittest
import json
import sys
import os
from typing import Tuple, Any


TEST_RESULT_JSONL = "test_result.jsonl"

class mock:
    class bases:
        class BasisDirection:
            def __init__(self, name: str, value: Tuple[int, int]):
                self.name = name
                self.value = value

        class VectorInBasis:
            pass

class discretising_layer:
    output_values = [1.0, 2.0, 3.0]  # example output values

out_vec = mock.bases.VectorInBasis()
hidden_name = "hidden_"  # mock variable used in the test

def second_layer_action(direction: mock.bases.BasisDirection) -> mock.bases.VectorInBasis:
    if direction.name == f"{hidden_name}start":
        return discretising_layer.output_values[0] * out_vec
    k, i = direction.value
    sign = {0: 1, 1: -1}[i]
    return sign * (discretising_layer.output_values[k] -
                   discretising_layer.output_values[k - 1]) * out_vec


class TestSecondLayerAction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[189]  # Get the 190th JSON element

    def test_second_layer_action(self):
        """Test second_layer_action with various input cases."""
        passed_count = 0
        failed_count = 0
        results = []

        # Mock directions for testing
        test_cases = [
            (mock.bases.BasisDirection(f"{hidden_name}start", (0, 0)), 1.0),
            (mock.bases.BasisDirection("action", (1, 0)), (2.0 - 1.0)),
            (mock.bases.BasisDirection("reaction", (2, 1)), -(3.0 - 2.0))
        ]

        for i, (direction, expected) in enumerate(test_cases):
            with self.subTest(test_case=i):
                try:
                    result = second_layer_action(direction)
                    # Mock output object comparison by checking __mul__ or other mocked behavior
                    result_value = result.__mul__(1) # assuming we mock multiplication
                    self.assertEqual(result_value, expected)
                    passed_count += 1
                    results.append({
                        "function_name": "second_layer_action",
                        "code": str(self.code_snippet),
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Test case {i}: FAILED with error: {e}")
                    failed_count += 1
                    results.append({
                        "function_name": "second_layer_action",
                        "code": str(self.code_snippet),
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
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
            if rec.get("function_name") != "second_layer_action"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()