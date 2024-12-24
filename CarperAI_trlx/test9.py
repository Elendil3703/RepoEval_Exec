import unittest
import json
import sys
import re
import os
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

_TRAINERS = {}

def register_class(cls, name):
    """Register a class under the specified name"""
    _TRAINERS[name] = cls
    setattr(sys.modules[__name__], name, cls)
    return cls

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[8]  # Get the 9th JSON element (index 8)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 9th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Static checks -------------------
                # 1) Check if _TRAINERS and register_class are defined in the code
                if "_TRAINERS" not in code:
                    print(f"Code snippet {i}: FAILED, '_TRAINERS' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "register_class",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "def register_class" not in code:
                    print(f"Code snippet {i}: FAILED, function 'register_class' not found.\n")
                    failed_count += 1
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
                    results.append({
                        "function_name": "register_class",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution and testing -------------------
                exec_globals = {
                    'sys': sys,
                    '_TRAINERS': {},  # Reset _TRAINERS for testing
                    'Any': Any,  # Inject Any,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if register_class exists in exec_locals
                    if 'register_class' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'register_class' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "register_class",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Use the register_class function to register classes
                    # Define TestTrainer and register it
                    exec('''
class TestTrainer:
    pass
register_class(TestTrainer, 'test_trainer')
''', exec_globals, exec_locals)

                    # Define OverwriteTrainer and register it to overwrite TestTrainer
                    exec('''
class OverwriteTrainer:
    pass
register_class(OverwriteTrainer, 'test_trainer')
''', exec_globals, exec_locals)

                    # Get the state of _TRAINERS after execution
                    _TRAINERS_after_exec = exec_globals['_TRAINERS']

                    # Fetch the OverwriteTrainer class from exec_locals
                    OverwriteTrainer = exec_locals.get('OverwriteTrainer') or exec_globals.get('OverwriteTrainer')

                    # Test: test_trainer should now be registered as OverwriteTrainer
                    self.assertIn(
                        "test_trainer",
                        _TRAINERS_after_exec,
                        f"Code snippet {i} did not correctly register 'test_trainer'.",
                    )
                    # Use assertIs to check class mapping correctly
                    self.assertIs(
                        _TRAINERS_after_exec["test_trainer"],
                        OverwriteTrainer,
                        f"Code snippet {i} did not map 'test_trainer' to the overwritten class.",
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

        # ============= Write results to test_result.jsonl =============
        # Read the existing test_result.jsonl (if exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name == "register_class"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "register_class_3"
        ]

        # Append the new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()