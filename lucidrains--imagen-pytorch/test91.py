import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCycleFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[90]  # Get the JSON element at index 90
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array at index 90")

    def test_code_snippets(self):
        """Dynamically test all code snippets for 'cycle'."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Static checks
                if "def cycle" not in code:
                    print(f"Code snippet {i}: FAILED, 'cycle' function definition not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "cycle",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if cycle is defined
                    if 'cycle' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'cycle' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "cycle",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Define a test dataset and expected cycle output
                    test_data = [1, 2, 3]
                    cycle_fn = exec_locals['cycle']
                    cycle_iterator = cycle_fn(test_data)
                    
                    # Collect a few results from the cycle iterator
                    output_sequence = [next(cycle_iterator) for _ in range(10)]
                    expected_sequence = test_data * 3 + [test_data[0]]

                    # Verify the cycling behavior
                    self.assertEqual(
                        output_sequence,
                        expected_sequence,
                        f"Code snippet {i} did not produce the expected cycle output.",
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "cycle",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "cycle",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

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
            if rec.get("function_name") != "cycle"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()