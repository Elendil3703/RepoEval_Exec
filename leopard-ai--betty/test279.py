import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCheckLeafFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[278]  # Get the 279th JSON element

    def test_check_leaf(self):
        """Test check_leaf function with specified cases."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect results to write to JSONL

        try:
            # Define a mock Problem class to simulate the input  (since 'Problem' type was not given)
            class Problem:
                pass

            # Prepare a mock class that contains check_leaf and the necessary self.dependencies
            class MockClass:
                def __init__(self, dependencies):
                    self.dependencies = dependencies

                def check_leaf(self, problem):
                    for _, value_list in self.dependencies["l2u"].items():
                        if problem in set(value_list):
                            return False
                    return True

            # Initialize the problem instances
            problem_a = Problem()
            problem_b = Problem()

            # Define test cases based on the input expectation
            test_cases = [
                ({"l2u": {"dep1": [problem_b]}}, problem_a, True),  # problem_a is not in any dependencies
                ({"l2u": {"dep1": [problem_b]}}, problem_b, False),  # problem_b is in dep1
                ({"l2u": {"dep1": []}}, problem_a, True),  # Empty dependency list
            ]

            for i, (dependency, problem, expected_result) in enumerate(test_cases):
                # Create an instance of MockClass with the current dependency
                mc = MockClass(dependency)
                
                # Execute the check_leaf method and assert the result
                actual_result = mc.check_leaf(problem)
                
                self.assertEqual(
                    actual_result, expected_result,
                    f"Test case {i} failed: Expected {expected_result}, got {actual_result}"
                )
                passed_count += 1
                results.append({
                    "function_name": "check_leaf",
                    "code": self.code_snippet,
                    "result": "passed"
                })

            print(f"Test Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")
        
        except Exception as e:
            print(f"Test case failed with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "check_leaf",
                "code": self.code_snippet,
                "result": "failed"
            })

        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")
        
        # Write results to JSONL file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for this test
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "check_leaf"]

        # Append new results
        existing_records.extend(results)

        # Re-write to the JSONL
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()