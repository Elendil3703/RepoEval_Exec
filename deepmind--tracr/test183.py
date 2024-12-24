import unittest
import json
import sys
import re
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"


class TestMakeNarySequenceMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[182]  # Get the 183rd JSON element (index 182)
        if not cls.code_snippet.strip():
            raise ValueError("Expected a non-empty code snippet at the specified index.")

    def test_make_nary_sequencemap(self):
        """Tests for the make_nary_sequencemap function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        code = self.code_snippet
        code_index = 182

        with self.subTest(code_index=code_index):
            print(f"Running test for code snippet {code_index}...")

            # ------------------- Static Checks -------------------
            # 1) Static check: Check if 'make_nary_sequencemap' is defined
            if "def make_nary_sequencemap" not in code:
                print(f"Code snippet {code_index}: FAILED, function 'make_nary_sequencemap' not found.\n")
                failed_count += 1
                results.append({
                    "function_name": "make_nary_sequencemap",
                    "code": code,
                    "result": "failed"
                })
                return

            # Regex to find function definitions
            func_pattern = r"def\s+make_nary_sequencemap\s*\("
            if not re.search(func_pattern, code):
                print(f"Code snippet {code_index}: FAILED, incorrect signature for 'make_nary_sequencemap'.\n")
                failed_count += 1
                results.append({
                    "function_name": "make_nary_sequencemap",
                    "code": code,
                    "result": "failed"
                })
                return

            # ------------------- Dynamic Execution and Tests -------------------

            # Mocking rasp.SequenceMap and rasp.Map
            class MockMap:
                def __init__(self, func, *args):
                    self.func = func
                    self.args = args

                def execute(self):
                    return [self.func(*a) for a in zip(*self.args)]

            class MockSequenceMap(MockMap):
                def __init__(self, func, values, sop):
                    super().__init__(func, values, sop)

            exec_globals = {
                'rasp': type('rasp', (object,), {'SequenceMap': MockSequenceMap, 'Map': MockMap}),
                'Any': Any,  # Injecting Any
            }
            exec_locals = {}

            try:
                # Execute the code snippet
                exec(code, exec_globals, exec_locals)

                if 'make_nary_sequencemap' not in exec_locals:
                    print(f"Code snippet {code_index}: FAILED, 'make_nary_sequencemap' not found in exec_locals.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "make_nary_sequencemap",
                        "code": code,
                        "result": "failed"
                    })
                    return

                # Testing the make_nary_sequencemap function
                def test_func(x, y, z):
                    return x + y + z

                sops = ([1, 2, 3], [4, 5, 6], [7, 8, 9])
                nary_map = exec_locals['make_nary_sequencemap'](test_func, *sops)
                result = nary_map.execute()

                expected_result = [12, 15, 18]
                self.assertEqual(
                    result,
                    expected_result,
                    f"Code snippet {code_index} did not produce expected output: {expected_result}",
                )

                print(f"Code snippet {code_index}: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "make_nary_sequencemap",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                print(f"Code snippet {code_index}: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "make_nary_sequencemap",
                    "code": code,
                    "result": "failed"
                })

        # ------------------- Writing to JSONL -------------------
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for make_nary_sequencemap
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "make_nary_sequencemap"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()