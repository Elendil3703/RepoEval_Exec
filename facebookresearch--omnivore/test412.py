import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestValidateOptimizerSchedulers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[411]  # Get the 412th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 412th JSON array")

    def test_validate_optimizer_schedulers(self):
        """Test the _validate_optimizer_schedulers function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Validate function definition -------------------
                if "def _validate_optimizer_schedulers" not in code:
                    print(f"Code snippet {i}: FAILED, function '_validate_optimizer_schedulers' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_validate_optimizer_schedulers",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution and testing -------------------
                exec_globals = {
                    'schedulers': None,
                    'optimizer': None,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Verify function presence
                    if '_validate_optimizer_schedulers' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_validate_optimizer_schedulers' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_validate_optimizer_schedulers",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Mock an optimizer with default options
                    mock_optimizer = type('MockOptimizer', (object,), {
                        'defaults': {'lr': 0.01, 'momentum': 0.9}
                    })()

                    # Test no schedulers scenario
                    exec_locals['schedulers'] = None
                    exec_locals['optimizer'] = mock_optimizer
                    exec_locals['_validate_optimizer_schedulers']()

                    # Test valid scheduler options
                    exec_locals['schedulers'] = [{'lr': 0.001}]
                    exec_locals['_validate_optimizer_schedulers']()

                    # Test invalid scheduler option
                    try:
                        exec_locals['schedulers'] = [{'invalid_option': 0.001}]
                        exec_locals['_validate_optimizer_schedulers']()
                        raise AssertionError("Expected an assertion error for invalid option, but none occurred.")
                    except AssertionError as e:
                        if "Optimizer option" not in str(e):
                            raise e

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_validate_optimizer_schedulers",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_validate_optimizer_schedulers",
                        "code": code,
                        "result": "failed"
                    })

        # Final test summary
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

        # Remove old records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_validate_optimizer_schedulers"
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