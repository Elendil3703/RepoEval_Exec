import unittest
import json
import os
from dataclasses import dataclass
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

# Mock classes to simulate the real environment
class Node:
    def __init__(self, model_block):
        self.attributes = {nodes.MODEL_BLOCK: model_block}

    def __getitem__(self, key):
        return self.attributes.get(key, None)

class nodes:
    MODEL_BLOCK = "model_block"

class transformers:
    class SeriesWithResiduals:
        def __init__(self, blocks):
            self.blocks = blocks

    class AttentionHead:
        pass

    class MultiAttentionHead:
        pass

    class MLP:
        pass

class TestNodeIsResidualBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[160]  # Get the 161st JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 161st JSON array")

    def test_node_is_residual_block(self):
        """Test the '_node_is_residual_block' function from the code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                exec_globals = {
                    'Node': Node,
                    'nodes': nodes,
                    'transformers': transformers,
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if '_node_is_residual_block' function is present
                    if '_node_is_residual_block' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_node_is_residual_block' function not found.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_node_is_residual_block",
                            "code": code,
                            "result": "failed"
                        })
                        continue
                    
                    _node_is_residual_block = exec_locals['_node_is_residual_block']

                    # Define test cases
                    test_cases = [
                        # Case 1: Valid residual block
                        (
                            Node(transformers.SeriesWithResiduals([
                                transformers.AttentionHead(),
                                transformers.MLP()
                            ])),
                            True
                        ),
                        # Case 2: Invalid residual block - Wrong block types
                        (
                            Node(transformers.SeriesWithResiduals([
                                transformers.MLP(),
                                transformers.AttentionHead()
                            ])),
                            False
                        ),
                        # Case 3: Invalid residual block - Non-series input
                        (
                            Node(None),
                            False
                        ),
                        # Case 4: Valid residual block with MultiAttentionHead
                        (
                            Node(transformers.SeriesWithResiduals([
                                transformers.MultiAttentionHead(),
                                transformers.MLP()
                            ])),
                            True
                        )
                    ]

                    # Run through test cases
                    for node, expected in test_cases:
                        result = _node_is_residual_block(node)
                        self.assertEqual(
                            result,
                            expected,
                            f"Code snippet {i}: expected {_node_is_residual_block} to return {expected} but got {result}."
                        )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_node_is_residual_block",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_node_is_residual_block",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
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

        # Remove old records for '_node_is_residual_block'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_node_is_residual_block"
        ]

        # Add new results
        existing_records.extend(results)

        # Write to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()