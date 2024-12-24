import unittest
import json
import os
from typing import Sequence

TEST_RESULT_JSONL = "test_result.jsonl"

class TestEvalAggregateFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[144]  # Get the 145th JSON element (index 144)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet at index 144")

    def test_eval_aggregate(self):
        """Test the eval_aggregate function in the provided code snippet."""
        exec_globals = {}
        exec_locals = {}

        # Dynamically execute the snippet code to define necessary classes and methods
        exec(self.code_snippet, exec_globals, exec_locals)

        # Check if eval_aggregate is defined
        self.assertIn('eval_aggregate', exec_locals, "eval_aggregate function not found in code snippet.")

        # Retrieve the eval_aggregate function
        eval_aggregate = exec_locals['eval_aggregate']

        # Mock objects and methods to test eval_aggregate
        class MockAggregate:
            def __init__(self, selector, sop, default):
                self.selector = selector
                self.sop = sop
                self.default = default

        def mock_evaluate(expression, xs):
            if expression == 'selector':
                return [[True, False], [False, True]]
            return xs

        def mock_mean(selected_values, default):
            return sum(selected_values) / len(selected_values) if selected_values else default

        def mock_get_selected(row, values):
            return [value for selected, value in zip(row, values) if selected]

        # Inject mock implementations
        exec_globals.update({
            '_mean': mock_mean,
            '_get_selected': mock_get_selected,
            'self': type('self', (), {'evaluate': mock_evaluate}),
        })

        # Test case
        sop = MockAggregate('selector', 'sop', 0)
        xs = [[1, 2], [3, 4]]

        expected_result = [1, 4]
        actual_result = eval_aggregate(exec_globals['self'], sop, xs)
        self.assertEqual(actual_result, expected_result, f"Expected {expected_result}, got {actual_result}")

        # Write results to JSONL
        results = [{
            "function_name": "eval_aggregate",
            "code": self.code_snippet,
            "result": "passed" if actual_result == expected_result else "failed"
        }]

        # Read existing results if any and filter out previous records for eval_aggregate
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove existing eval_aggregate records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "eval_aggregate"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()