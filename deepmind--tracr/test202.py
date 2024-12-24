import unittest
import json
import sys
import os
import numpy as np
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class VectorSpaceWithBasis:
    def __init__(self, dimension):
        self.dimension = dimension

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[201]  # Get the 202nd JSON element
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_init_method(self):
        """Dynamically test __init__ method in the code snippet with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        print(f"Running test for provided code snippet...")

        # ------------------- Static checks -------------------
        # Check for class and __init__
        if "class " not in self.code_snippet:
            print(f"Code snippet: FAILED, class definition not found in code.\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": self.code_snippet,
                "result": "failed"
            })
            return

        if "def __init__" not in self.code_snippet:
            print(f"Code snippet: FAILED, function '__init__' not found.\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": self.code_snippet,
                "result": "failed"
            })
            return

        # ------------------- Dynamic execution -------------------
        exec_globals = {
            'np': np,
            'VectorSpaceWithBasis': VectorSpaceWithBasis,
            'Any': Any
        }
        exec_locals = {}

        try:
            # Dynamically execute code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check presence of class
            class_name_pattern = r"class\s+(\w+)"
            class_name_match = re.search(class_name_pattern, self.code_snippet)
            assert class_name_match, "Cannot find class name."

            class_name = class_name_match.group(1)

            assert class_name in exec_locals, f"Class {class_name} not found after exec."

            # Initialize the class with sample inputs
            vector_space = VectorSpaceWithBasis(3)
            matrix = np.array([[1, 2], [3, 4], [5, 6]])

            try:
                obj = exec_locals[class_name](vector_space, vector_space, matrix)
                self.assertEqual(obj.input_space, vector_space, "input_space not set correctly.")
                self.assertEqual(obj.output_space, vector_space, "output_space not set correctly.")
                np.testing.assert_array_equal(obj.matrix, matrix, "matrix not set correctly.")

                print(f"Code snippet: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "__init__",
                    "code": self.code_snippet,
                    "result": "passed"
                })
            except Exception as exc:
                print(f"Code snippet: FAILED during object initialization with error: {exc}\n")
                failed_count += 1
                results.append({
                    "function_name": "__init__",
                    "code": self.code_snippet,
                    "result": "failed"
                })

        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Final statistics
        total_tests = passed_count + failed_count
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {total_tests}\n")
        self.assertEqual(total_tests, 1, "Test count mismatch!")

        # ============= Write results to test_result.jsonl =============
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
            if rec.get("function_name") != "__init__"
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