import unittest
import json
import sys
import re
import os
import logging
from unittest.mock import patch
from io import StringIO

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        # Get the 11th JSON element (index 10)
        cls.code_snippets = data[10]
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 11th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets for 'ault_log_level' function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Static checks -------------------
                # 1) Check if 'ault_log_level' is defined in code
                if "ault_log_level" not in code:
                    print(f"Code snippet {i}: FAILED, 'ault_log_level' not found in code.\n")
                    failed_count += 1
                    # Record the failure
                    results.append({
                        "function_name": "ault_log_level",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+ault_log_level\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'ault_log_level'.\n")
                    failed_count += 1
                    # Record the failure
                    results.append({
                        "function_name": "ault_log_level",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution and logic testing -------------------
                exec_globals = {
                    'os': os,
                    'logging': logging,
                    'log_levels': {
                        'critical': logging.CRITICAL,
                        'error': logging.ERROR,
                        'warning': logging.WARNING,
                        'info': logging.INFO,
                        'debug': logging.DEBUG,
                        'notset': logging.NOTSET,
                    },
                    '_default_log_level': logging.WARNING,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'ault_log_level' exists after execution
                    if 'ault_log_level' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'ault_log_level' not found after execution.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "ault_log_level",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Get the 'ault_log_level' function
                    ault_log_level = exec_locals['ault_log_level']

                    # Define test cases: (TRLX_VERBOSITY value, expected result)
                    test_cases = [
                        (None, exec_globals['_default_log_level']),
                        ('debug', logging.DEBUG),
                        ('info', logging.INFO),
                        ('warning', logging.WARNING),
                        ('error', logging.ERROR),
                        ('critical', logging.CRITICAL),
                        ('notset', logging.NOTSET),
                        ('INVALID_LEVEL', exec_globals['_default_log_level']),
                    ]

                    for env_value, expected in test_cases:
                        with patch.dict('os.environ', {'TRLX_VERBOSITY': env_value} if env_value is not None else {}, clear=False):
                            # Set up logging capture
                            log_stream = StringIO()
                            handler = logging.StreamHandler(log_stream)
                            logger = logging.getLogger()
                            logger.addHandler(handler)

                            try:
                                result = ault_log_level()
                            finally:
                                logger.removeHandler(handler)

                            log_contents = log_stream.getvalue()

                            if env_value is None or env_value.lower() in exec_globals['log_levels']:
                                self.assertEqual(
                                    result,
                                    expected,
                                    f"Expected log level {expected} for TRLX_VERBOSITY={env_value}, got {result}"
                                )

                                # There should be no warnings
                                self.assertEqual(
                                    log_contents.strip(),
                                    '',
                                    f"Unexpected warnings for TRLX_VERBOSITY={env_value}"
                                )
                            else:
                                # Should log a warning and return default
                                self.assertEqual(
                                    result,
                                    exec_globals['_default_log_level'],
                                    f"Expected default log level for invalid TRLX_VERBOSITY={env_value}"
                                )
                                self.assertNotEqual(
                                    log_contents.strip(),
                                    '',
                                    f"Expected warnings for invalid TRLX_VERBOSITY={env_value}"
                                )
                                self.assertIn(
                                    'Unknown option TRLX_VERBOSITY',
                                    log_contents,
                                    "Warning message not generated for invalid TRLX_VERBOSITY"
                                )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "ault_log_level",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "ault_log_level",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "ault_log_level"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "ault_log_level"
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