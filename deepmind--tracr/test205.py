import unittest
import json
import os
from typing import Sequence

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCombineInParallelResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and extract the 205th snippet (index 204)
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[204]  # Extract the 205th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 205th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Static check: Verify the presence of combine_in_parallel
                if "def combine_in_parallel" not in code:
                    print(f"Code snippet {i}: FAILED, 'combine_in_parallel' not defined.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "combine_in_parallel",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Execute dynamic tests
                exec_globals = {}
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check for the presence of 'combine_in_parallel'
                    if 'combine_in_parallel' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'combine_in_parallel' not found.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "combine_in_parallel",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Mock class and dependencies for testing
                    class MockSpace:
                        def join_vector_spaces(*args):
                            return MockSpace()

                        def null_vector(self):
                            return 0

                    class MockBasisDirection:
                        pass

                    class MockVectorInBasis:
                        pass

                    class MockLinear:
                        input_space = MockSpace()
                        output_space = MockSpace()

                        def __call__(self, vec):
                            return MockVectorInBasis()

                    combine_in_parallel = exec_locals['combine_in_parallel']

                    # Create mock functions
                    fns = [MockLinear(), MockLinear()]

                    # Test: combine_in_parallel with mocks
                    result = combine_in_parallel(MockLinear, fns)

                    # Check the result is of the correct type
                    self.assertIsInstance(
                        result,
                        MockLinear,
                        f"Code snippet {i} did not return an instance of the expected class."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "combine_in_parallel",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "combine_in_parallel",
                        "code": code,
                        "result": "failed"
                    })

        # Summary report
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
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
            if rec.get("function_name") != "combine_in_parallel"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()