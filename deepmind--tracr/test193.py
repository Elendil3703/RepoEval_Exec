import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

def mock(direction_name, value):
    """A mock function to simulate the direction object behavior."""
    class MockDirection:
        def __init__(self):
            self.name = direction_name
            self.value = value
    return MockDirection()

def prepare_global_vars():
    """Prepare and return the global variables needed for the function to execute."""
    class MockBases:
        class BasisDirection:
            pass
        
        class VectorInBasis:
            pass
            
        def null_vector(self):
            return "NULL_VECTOR"  # Assuming this is a stand-in for an actual null vector representation

    return {
        'bases': MockBases,
        'hidden_name': 'h',
        'input1_factor': 2,
        'input2_factor': 3,
        'out_vec': 10,
        'output_space': MockBases()
    }

class TestSecondLayerAction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        try:
            cls.code_snippet = data[192]  # Get the 193rd JSON element (index 192)
        except IndexError:
            raise ValueError("Expected the 193rd JSON element not found in RepoEval_result.json")

    def test_second_layer_action(self):
        """Dynamically test the second_layer_action function with multiple scenarios."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        code = self.code_snippet
        exec_globals = prepare_global_vars()
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Ensure the function `second_layer_action` exists
            if 'second_layer_action' not in exec_locals:
                raise ValueError("'second_layer_action' function not defined in the code snippet")

            second_layer_action = exec_locals['second_layer_action']

            # Test scenarios
            test_cases = [
                {"direction_name": "hx", "value": 1, "expected": 20},
                {"direction_name": "hy", "value": 1, "expected": 30},
                {"direction_name": "hz", "value": 1, "expected": "NULL_VECTOR"},
            ]

            for i, test in enumerate(test_cases):
                with self.subTest(test_index=i):
                    mock_direction = mock(test['direction_name'], test['value'])
                    result = second_layer_action(mock_direction)

                    # Assert the correct output
                    self.assertEqual(
                        result,
                        test['expected'],
                        f"Test case {i} failed: expected {test['expected']}, got {result}"
                    )

                    print(f"Test case {i}: PASSED")
                    passed_count += 1

            results.append({
                "function_name": "second_layer_action",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Function test failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "second_layer_action",
                "code": code,
                "result": "failed",
                "error": str(e)
            })

        # Final summary
        print(f"Test Summary: {passed_count} passed, {failed_count} failed")

        # ============= Write results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if it doesn't exist)
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
            if rec.get("function_name") != "second_layer_action"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()