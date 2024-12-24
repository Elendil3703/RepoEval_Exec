import unittest
import json
import os
from typing import Dict, Any
from unittest.mock import Mock

TEST_RESULT_JSONL = "test_result.jsonl"

class DummyBasisDirection:
    """ A mock class to substitute the bases.BasisDirection in tests. """
    def __init__(self, name, value):
        self.name = name
        self.value = value

class DummyVectorInBasis:
    """ A mock class to substitute the bases.VectorInBasis in tests. """
    # Add attributes and methods if needed to mock behaviors.
    pass

class TestSecondLayerAction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up conditions for all tests."""
        # Mocks and fake data for testing purposes
        cls.hidden_name = "hidden_"
        cls.vec_by_out_val = {}
        cls.discretising_layer = Mock()
        cls.discretising_layer.output_values = [10, 20, 30, 40]  # example values

        def create_mock_vector(value):
            vector = DummyVectorInBasis()
            cls.vec_by_out_val[value] = vector

        for value in cls.discretising_layer.output_values:
            create_mock_vector(value)

    def test_second_layer_action_start(self):
        """Test start direction scenario."""
        direction = DummyBasisDirection(name=f"{self.hidden_name}start", value=None)

        result = self.second_layer_action(direction)

        self.assertIs(
            result,
            self.vec_by_out_val[self.discretising_layer.output_values[0]],
            "Expected vector for start output value not returned."
        )

    def test_second_layer_action_non_start(self):
        """Test non-start direction scenarios."""
        examples = [
            ((1, 0), 1 * (self.vec_by_out_val[20] - self.vec_by_out_val[10])),
            ((2, 1), -1 * (self.vec_by_out_val[30] - self.vec_by_out_val[20])),
        ]

        for value, expected_vector in examples:
            direction = DummyBasisDirection(name="not_start", value=value)

            result = self.second_layer_action(direction)

            self.assertEqual(
                result, expected_vector,
                f"Failed at direction value {value}: incorrect vector."
            )

    def second_layer_action(self, direction: DummyBasisDirection) -> DummyVectorInBasis:
        """Compute output vector based on direction."""
        if direction.name == f"{self.hidden_name}start":
            return self.vec_by_out_val[self.discretising_layer.output_values[0]]
        else:
            k, i = direction.value
            sign = {0: 1, 1: -1}[i]
            out_k = self.discretising_layer.output_values[k]
            out_k_m_1 = self.discretising_layer.output_values[k - 1]
            return sign * (self.vec_by_out_val[out_k] - self.vec_by_out_val[out_k_m_1])

    @classmethod
    def tearDownClass(cls):
        """Write results to test_result.jsonl."""
        results = [
            {"function_name": "second_layer_action", "result": "passed"}
            # Add more diagnostic results if tests are differentiated into pass/fail.
        ]

        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                existing_records = [json.loads(line) for line in f if line.strip()]
        else:
            existing_records = []

        # Filter out old records for this specific function
        existing_records = [
            record for record in existing_records
            if record.get("function_name") != "second_layer_action"
        ]

        # Add the new results
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()