import unittest
import json
import sys
import os
import logging
from threading import Lock

TEST_RESULT_JSONL = "test_result.jsonl"

_lock = Lock()
_default_handler = None

class TestConfigureLibraryRootLogger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[11]  # Get the 12th JSON element (0-indexed)
        if not cls.code_snippet:
            raise ValueError("Expected a valid code snippet in the JSON data")

    def test_configure_library_root_logger(self):
        """Test the _configure_library_root_logger function dynamically."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        code = self.code_snippet
        print("Testing the _configure_library_root_logger function...")

        # ------------------- Dynamic Execution -------------------
        exec_globals = {
            '_lock': Lock(),
            '_default_handler': None,
            'logging': logging,
            'sys': sys,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if _configure_library_root_logger is defined
            if '_configure_library_root_logger' not in exec_locals:
                print("FAILED: '_configure_library_root_logger' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "_configure_library_root_logger",
                    "code": code,
                    "result": "failed"
                })
                return

            _configure_library_root_logger = exec_locals['_configure_library_root_logger']

            # Reset global variables for testing
            exec_globals['_default_handler'] = None

            # Test 1: Call the function and check if a handler is configured
            _configure_library_root_logger()
            self.assertIsNotNone(
                exec_globals['_default_handler'],
                "Default handler was not set after calling _configure_library_root_logger"
            )
            self.assertIsInstance(
                exec_globals['_default_handler'], logging.StreamHandler,
                "Default handler is not an instance of logging.StreamHandler"
            )

            # Test 2: Call the function again and ensure it does not reconfigure
            handler_before = exec_globals['_default_handler']
            _configure_library_root_logger()
            self.assertEqual(
                handler_before,
                exec_globals['_default_handler'],
                "Handler should not be replaced when _configure_library_root_logger is called again"
            )

            # Test 3: Validate logger configuration
            library_root_logger = logging.getLogger()  # Root logger
            self.assertIn(
                handler_before,
                library_root_logger.handlers,
                "Default handler not found in library root logger handlers"
            )
            self.assertFalse(
                library_root_logger.propagate,
                "Library root logger should have propagate set to False"
            )

            print("PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "_configure_library_root_logger",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "_configure_library_root_logger",
                "code": code,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= Write Test Results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for _configure_library_root_logger
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_configure_library_root_logger"
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