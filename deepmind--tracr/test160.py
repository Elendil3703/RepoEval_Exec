import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestNodeIsAttn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[159]  # Get the 160th JSON element (index 159)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet for testing")

    def test_node_is_attn(self):
        """Test the _node_is_attn function from the code snippets dynamically."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "_node_is_attn" not in code:
                    print(f"Code snippet {i}: FAILED, '_node_is_attn' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_node_is_attn",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {}
                exec_locals = {
                    'nodes': {
                        'MODEL_BLOCK': 'model_block_key'  # Stub for nodes.MODEL_BLOCK
                    },
                    'transformers': __import__('types'),  # Use the types module as a stub
                    'Node': dict,  # Stub for Node
                }

                try:
                    exec(code, exec_globals, exec_locals)

                    # Check if the function exists
                    if '_node_is_attn' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_node_is_attn' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_node_is_attn",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    _node_is_attn = exec_locals['_node_is_attn']

                    # Test cases
                    attention_node = {'model_block_key': exec_globals['type']('AttentionHead', (), {})}
                    multi_attention_node = {'model_block_key': exec_globals['type']('MultiAttentionHead', (), {})}
                    non_attention_node = {'model_block_key': exec_globals['type']('NonAttention', (), {})}
                    non_model_block_node = {}

                    self.assertTrue(_node_is_attn(attention_node),
                                    f"Code snippet {i} failed to identify an AttentionHead node.")

                    self.assertTrue(_node_is_attn(multi_attention_node),
                                    f"Code snippet {i} failed to identify a MultiAttentionHead node.")

                    self.assertFalse(_node_is_attn(non_attention_node),
                                     f"Code snippet {i} incorrectly identified a non-attention node as attention.")

                    self.assertFalse(_node_is_attn(non_model_block_node),
                                     f"Code snippet {i} incorrectly identified a node without model_block as attention.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_node_is_attn",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_node_is_attn",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
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
            if rec.get("function_name") != "_node_is_attn"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()