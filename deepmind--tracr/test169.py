import unittest
import json
import os
import sys
import re
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMakeInputSpaceFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[168]  # Get the 169th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_make_input_space(self):
        """Dynamically test the _make_input_space function implementation."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                if "_make_input_space" not in code:
                    print(f"Code snippet {i}: FAILED, '_make_input_space' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_make_input_space",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+_make_input_space\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for '_make_input_space'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_make_input_space",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'sys': sys,
                    'bases': type('bases', (), {})(),  # Mocking 'bases' to simulate the behavior
                }
                exec_locals = {}

                # Mocking necessary components of 'bases' for test purposes
                class MockVectorSpaceWithBasis:
                    @staticmethod
                    def from_values(name, values):
                        return {name: list(values)}

                    @staticmethod
                    def from_names(names):
                        return {name: None for name in names}
                
                def mock_join_vector_spaces(*spaces):
                    result_space = {}
                    for space in spaces:
                        result_space.update(space)
                    return result_space

                setattr(exec_globals['bases'], 'VectorSpaceWithBasis', MockVectorSpaceWithBasis)
                setattr(exec_globals['bases'], 'join_vector_spaces', mock_join_vector_spaces)

                try:
                    exec(code, exec_globals, exec_locals)

                    if '_make_input_space' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_make_input_space' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_make_input_space",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    _make_input_space_fn = exec_locals['_make_input_space']

                    vocab = ['a', 'b', 'c']
                    max_seq_len = 3
                    input_space = _make_input_space_fn(vocab, max_seq_len)

                    expected_space = {
                        'tokens': vocab,
                        'indices': [0, 1, 2],
                        _ONE_DIRECTION: None,
                        _BOS_DIRECTION: None,
                    }

                    self.assertEqual(input_space, expected_space, f"Code snippet {i} failed space equality check.")
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_make_input_space",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_make_input_space",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

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
            if rec.get("function_name") != "_make_input_space"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()