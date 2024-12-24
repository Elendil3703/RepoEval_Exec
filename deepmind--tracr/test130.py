import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[129]  # Get the 130th JSON element (index 129)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 130th JSON array")

    def test_init_function(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "def __init__(" not in code:
                    print(f"Code snippet {i}: FAILED, '__init__' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'Any': Any
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    if '__init__' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '__init__' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    SOp = type('SOp', (object,), {})
                    class TestClass(exec_locals['TestClass']):
                        pass

                    instance = TestClass(SOp(), SOp(), 1.0, 2.0)
                    self.assertEqual(instance.fst_fac, 1.0, f"Code snippet {i} did not correctly assign 'fst_fac'")
                    self.assertEqual(instance.snd_fac, 2.0, f"Code snippet {i} did not correctly assign 'snd_fac'")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(
            passed_count + failed_count, len(self.code_snippets),
            "Test count mismatch!"
        )

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
            if rec.get("function_name") != "__init__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()