import unittest
import json
import sys
import os
from typing import Any, Dict

TEST_RESULT_JSONL = "test_result.jsonl"
_DEFAULT_NAME_BY_CLASS = {}

class RASPExpr:
    """A placeholder class for RASPExpr used for testing."""
    pass

def default_name(expr: RASPExpr) -> Dict[str, str]:
    for cls, name in _DEFAULT_NAME_BY_CLASS.items():
        if isinstance(expr, cls):
            return name

    raise NotImplementedError(f"{expr} was not given a default name!")

class TestDefaultNameFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the RepoEval_result.json file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[139]  # Get the 140th JSON element (index 139)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 140th JSON array")

    def test_code_snippets(self):
        """Dynamically test the code snippets in the JSON with default_name function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Static Checks -------------------
                if "_DEFAULT_NAME_BY_CLASS" not in code:
                    print(f"Code snippet {i}: FAILED, '_DEFAULT_NAME_BY_CLASS' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "default_name",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "def default_name" not in code:
                    print(f"Code snippet {i}: FAILED, function 'default_name' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "default_name",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Testing -------------------
                exec_globals = {
                    '_DEFAULT_NAME_BY_CLASS': {},
                    'RASPExpr': RASPExpr,
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if default_name function exists
                    if 'default_name' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'default_name' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "default_name",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Register a test class and name
                    class TestExpr(RASPExpr):
                        pass

                    test_name = "test_name"
                    exec_globals['_DEFAULT_NAME_BY_CLASS'][TestExpr] = test_name

                    # Test the default_name function
                    self.assertEqual(
                        exec_locals['default_name'](TestExpr()),
                        test_name,
                        f"Code snippet {i} did not return the correct name for TestExpr."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "default_name",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "default_name",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if not exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records with function_name == "default_name"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "default_name"
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