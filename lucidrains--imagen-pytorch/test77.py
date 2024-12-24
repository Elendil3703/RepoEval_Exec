import unittest
import json
import os
import sys
import re
from typing import Any  # Ensure Any is available in the injected environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthForwardResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[76]  # Get the 77th JSON element (index 76)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test code snippets related to forward function."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Validity Checks -------------------
                if "def forward" not in code:
                    print(f"Code snippet {i}: FAILED, 'forward' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+forward\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'forward'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Testing -------------------
                exec_globals = {
                    'sys': sys,
                    'Any': Any,  # Inject Any
                }
                exec_locals = {}

                try:
                    # Define auxiliary functions used in the forward function
                    def exists(val):
                        return val is not None

                    def rearrange(x, pattern):
                        # Dummy rearrange implementation for testing
                        return x

                    def pack(tensors, pattern):
                        # Dummy pack implementation for testing
                        return tensors[0], None

                    def unpack(tensor, shapes, pattern):
                        # Dummy unpack implementation for testing
                        return tensor,

                    exec_globals['exists'] = exists
                    exec_globals['rearrange'] = rearrange
                    exec_globals['pack'] = pack
                    exec_globals['unpack'] = unpack

                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if forward function defined
                    if 'forward' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'forward' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Simulate the forward function's behavior for testing
                    class MockModel:
                        def block1(self, x):
                            return x  # Dummy block1 for testing

                        def block2(self, x, scale_shift=None):
                            return x  # Dummy block2 for testing

                        def gca(self, x):
                            return x  # Dummy gca for testing

                        def res_conv(self, x):
                            return x  # Dummy res_conv for testing

                    mock_model = MockModel()
                    forward_func = exec_locals['forward'].__get__(mock_model)

                    # Test with dummy inputs
                    x = "dummy_input"
                    time_emb = "dummy_time_emb"
                    cond = "dummy_cond"

                    result = forward_func(x, time_emb, cond)
                    expected_result = "dummy_result"  # Expectation based on dummy implementation

                    # Compare results
                    self.assertEqual(result, expected_result, f"Code snippet {i} failed with unexpected output.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
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

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "forward"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()