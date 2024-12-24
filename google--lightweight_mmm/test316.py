import unittest
import json
import os
from typing import Any  # Ensure Any is injected

TEST_RESULT_JSONL = "test_result.jsonl"

def hill(data, half_max_effective_concentration, slope):
    """Implements the Hill function for adstock transformation."""
    # Assuming the core logic of the hill function here
    import jax.numpy as jnp
    return 1 / (1 + (half_max_effective_concentration / data)**slope)

class TestHillFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code = data[315]  # Get the 316th JSON element (index 315)
        if not cls.code:
            raise ValueError("Expected code snippet at index 315")

    def test_hill_function(self):
        """Test the hill function logic with sample inputs and outputs."""
        # Sample test cases for hill function
        test_cases = [
            {
                "data": [10, 20, 30],
                "half_max_effective_concentration": 15,
                "slope": 2,
                "expected": [0.04, 0.5, 0.9]  # Example expected output
            },
            {
                "data": [1, 5, 10],
                "half_max_effective_concentration": 5,
                "slope": 1,
                "expected": [0.17, 0.5, 0.67]  # Example expected output
            }
        ]

        passed_count = 0
        failed_count = 0
        results = []

        for i, test_case in enumerate(test_cases):
            with self.subTest(test_index=i):
                print(f"Running test for hill function test case {i}...")
                try:
                    # Execute the test
                    result = hill(
                        data=test_case["data"],
                        half_max_effective_concentration=test_case["half_max_effective_concentration"],
                        slope=test_case["slope"]
                    )
                    # Evaluate the result against expected values
                    self.assertTrue(
                        (result - test_case["expected"]) < 0.1,
                        f"Test case {i} failed. Expected {test_case['expected']}, got {result}"
                    )
                    print(f"Test case {i}: PASSED.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "hill",
                        "test_index": i,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Test case {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "hill",
                        "test_index": i,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # ============= Write Results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "hill"
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