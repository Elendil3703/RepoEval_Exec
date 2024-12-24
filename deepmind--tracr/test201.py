import unittest
import json
import os
import sys

TEST_RESULT_JSONL = "test_result.jsonl"


class TestOperationFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[200]  # Get the 201st JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON data")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with checks specific to operation_fn."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to be written into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static checks
                if "def operation_fn" not in code:
                    print(f"Code snippet {i}: FAILED, 'operation_fn' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "operation_fn",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Check function signature
                func_pattern = r"def\s+operation_fn\s*\(direction\):"
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'operation_fn'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "operation_fn",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and testing
                exec_globals = {
                    'sys': sys,
                    'output_space': type('MockOutputSpace', (object,), {
                        'vector_from_basis_direction': lambda d: f"vector-{d}",
                        'null_vector': lambda: "null_vector"
                    })(),
                    'from_hidden': lambda d: (f"dir1-{d}", f"dir2-{d}"),
                    'operation': lambda d1, d2: f"{d1}-{d2}"
                }
                exec_locals = {}

                try:
                    # Execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if operation_fn is defined
                    if 'operation_fn' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'operation_fn' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "operation_fn",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test operation_fn logic
                    operation_fn = exec_locals['operation_fn']

                    output_space_before_exec = exec_globals['output_space']

                    # Execute and validate operation_fn
                    direction = 'test-direction'
                    expected_output = "vector-dir1-test-direction-dir2-test-direction"
                    self.assertEqual(
                        operation_fn(direction),
                        expected_output,
                        f"Code snippet {i} did not return expected output."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "operation_fn",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "operation_fn",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
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

        # Remove old records for operation_fn
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "operation_fn"
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