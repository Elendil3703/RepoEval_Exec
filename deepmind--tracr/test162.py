import unittest
import json
import os
from typing import Sequence  # Ensure Sequence is available in the injected environment.

TEST_RESULT_JSONL = "test_result.jsonl"

class Node:
    def __init__(self, is_attn: bool):
        self.is_attn = is_attn

def _node_is_attn(node: Node) -> bool:
    """Mock function to check if a node is an attention node."""
    return node.is_attn

class TestAllAttnNodesFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.test_data = data[161]  # Get the 162nd JSON element
        if len(cls.test_data) < 1:
            raise ValueError("Expected at least one code snippet in the 162nd JSON array.")

    def test_all_attn_nodes(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.test_data):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Check for the presence of '_all_attn_nodes' in the code.
                if "_all_attn_nodes" not in code:
                    print(f"Code snippet {i}: FAILED, '_all_attn_nodes' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_all_attn_nodes",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'Sequence': Sequence,
                    '_node_is_attn': _node_is_attn,
                    'Node': Node,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure _all_attn_nodes exists
                    if '_all_attn_nodes' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_all_attn_nodes' not found after exec.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_all_attn_nodes",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    _all_attn_nodes = exec_locals['_all_attn_nodes']

                    # Test cases
                    test_cases = [
                        ([], True),  # Empty list
                        ([Node(True), Node(True)], True),  # All attention nodes
                        ([Node(False), Node(True)], False),  # Not all attention nodes
                        ([Node(False)], False),  # Single non-attention node
                        ([Node(True)], True)  # Single attention node
                    ]

                    for test_index, (nodes, expected) in enumerate(test_cases):
                        with self.subTest(test_case_index=test_index):
                            result = _all_attn_nodes(nodes)
                            self.assertEqual(
                                result, expected,
                                f"Test case {test_index} failed for code snippet {i}."
                            )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_all_attn_nodes",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_all_attn_nodes",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.test_data)}\n")
        self.assertEqual(passed_count + failed_count, len(self.test_data), "Test count mismatch!")

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for _all_attn_nodes
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_all_attn_nodes"
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