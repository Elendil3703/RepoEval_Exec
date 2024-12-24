import unittest
import json
import os
import numpy as np  # Ensure numpy is imported for numerical operations

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[377]  # Get the 378th JSON element (0-based index)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the selected JSON array")

    def test_code_snippets(self):
        """Dynamically test the _reduce function logic."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static checks -------------------
                if "def _reduce" not in code:
                    print(f"Code snippet {i}: FAILED, '_reduce' function not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_reduce",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution and logic testing -------------------
                exec_globals = {'np': np}
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if _reduce is actually defined
                    if '_reduce' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_reduce' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_reduce",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the _reduce function logic
                    _reduce = exec_locals['_reduce']

                    # Create a test input
                    test_features = np.random.rand(10, 3, 4, 4)  # Sample input array

                    # Calculate expected result using numpy operations
                    expected_output = test_features.reshape(10, 3, -1).mean(axis=-1)

                    # Call the _reduce function and get the result
                    actual_output = _reduce(test_features)

                    # Assertions to check if actual output matches expected output
                    np.testing.assert_allclose(
                        actual_output,
                        expected_output,
                        err_msg=f"Code snippet {i} returned incorrect result.",
                        rtol=1e-5,  # Tolerance set for floating point comparison
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_reduce",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_reduce",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if it doesn't exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "_reduce"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_reduce"
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