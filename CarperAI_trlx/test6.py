import unittest
import json
import sys
import re
import os
from typing import Any  # Ensure Any is injected into the environment

TEST_RESULT_JSONL = "test_result.jsonl"

_METHODS = {}

def register_class(cls, name):
    """Function used to register a method config
    Args:
        cls: Class to be registered
        name: Name of the method
    """
    _METHODS[name] = cls
    setattr(sys.modules[__name__], name, cls)
    return cls

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[5]  # Get the 6th JSON element (index 5)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 6th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Static checks -------------------
                # 1) Check if _METHODS is defined and register_class exists in the snippet
                if "_METHODS" not in code:
                    print(f"Code snippet {i}: FAILED, '_METHODS' not found in code.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "register_class",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "def register_class" not in code:
                    print(f"Code snippet {i}: FAILED, function 'register_class' not found.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "register_class",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+register_class\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'register_class'.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "register_class",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution and logic checks -------------------
                exec_globals = {
                    'sys': sys,
                    '_METHODS': _METHODS,  # Ensure the _METHODS is in the global scope for exec()
                    'Any': Any,  # Inject Any
                    'register_class': register_class,  # Explicitly inject register_class function
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if register_class is present in the executed locals
                    if 'register_class' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'register_class' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "register_class",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Define classes outside the register_class decorator scope
                    class TestClass:
                        pass

                    # Use register_class to test method registration logic
                    register_class(TestClass, "test_class")

                    # Try registering again to see if it overrides
                    class OverwriteClass:
                        pass

                    # Register the class again with the same name
                    register_class(OverwriteClass, "test_class")

                    # Get the resulting _METHODS after execution
                    _METHODS_after_exec = exec_globals['_METHODS']

                    # Test: test_class should now be registered as OverwriteClass
                    self.assertIn(
                        "test_class",
                        _METHODS_after_exec,
                        f"Code snippet {i} did not correctly register 'test_class'.",
                    )
                    self.assertEqual(
                        _METHODS_after_exec["test_class"],
                        OverwriteClass,
                        f"Code snippet {i} did not map 'test_class' to the overwritten class.",
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "register_class",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "register_class",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write the test results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records where function_name == "register_class"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "register_class_2"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()