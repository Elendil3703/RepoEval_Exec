import unittest
import json
import os
from unittest.mock import Mock

TEST_RESULT_JSONL = "test_result.jsonl"

class TestLogFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[271]  # Get the 272nd JSON element (index 271)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 272nd JSON array")

    def test_log_snippets(self):
        """Test code snippet containing 'log' function."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Basic static check: ensure 'def log' is in the code
                if "def log" not in code:
                    print(f"Code snippet {i}: FAILED, 'log' function not found in the code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "log",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {}
                exec_locals = {}

                try:
                    # Execute the code snippet to define the log function
                    exec(code, exec_globals, exec_locals)

                    if 'log' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'log' function not found in exec_locals after exec.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "log",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Mocking objects required for testing
                    logger_mock = Mock()
                    log_instance = Mock()
                    log_instance.logger = logger_mock
                    log_instance._name = "TestProblem"
                    log_instance._count = 42
                    log_instance.log_local_step = False
                    
                    # Retrieve the function to test
                    log_function = exec_locals['log']

                    # Execute the log function
                    stats = {'loss': 0.25, 'accuracy': 0.8}
                    global_step = 100

                    # Test logging with global step
                    log_function(log_instance, stats, global_step)

                    # Verify logging behaviour
                    log_instance.logger.info.assert_called_once_with(
                        '[Problem "TestProblem"] [Global Step 100] [Local Step 42] log_from_loss_dict(stats)'
                    )
                    log_instance.logger.log.assert_called_once_with(
                        stats, tag="TestProblem", step=global_step
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "log",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "log",
                        "code": code,
                        "result": "failed"
                    })

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

        # Remove old records for the 'log' function
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "log"]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()