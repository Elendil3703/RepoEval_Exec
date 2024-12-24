import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestRepoEvalResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[155]  # Get the 156th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 156th JSON array")

    def test_code_snippets(self):
        """Test _get_input_space_from_node functionality in the JSON."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Check for presence of _get_input_space_from_node
                if "_get_input_space_from_node" not in code:
                    print(f"Code snippet {i}: FAILED, '_get_input_space_from_node' not found in code.\n")
                    failed_count += 1
                    # Save failed record
                    results.append({
                        "function_name": "_get_input_space_from_node",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {'Any': Any, 'transformers': transformers, 'nodes': nodes, 'bases': bases}
                exec_locals = {}

                try:
                    # Dynamically execute code
                    exec(code, exec_globals, exec_locals)

                    # Check if _get_input_space_from_node was defined
                    if '_get_input_space_from_node' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_get_input_space_from_node' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_get_input_space_from_node",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Define a mock node for testing
                    class MockMLP:
                        class FST:
                            input_space = "mlp_input_space"
                        fst = FST()
                    
                    class MockAttentionHead:
                        class W_QK:
                            left_space = "left_space"
                            right_space = "right_space"
                        class W_OV:
                            input_space = "attention_input_space"
                        
                        w_qk = W_QK()
                        w_ov = W_OV()
                    
                    class Nodes:
                        MODEL_BLOCK = "model_block"

                    exec_globals['nodes'] = Nodes
                    node_mlp = {Nodes.MODEL_BLOCK: MockMLP()}
                    node_attention = {Nodes.MODEL_BLOCK: MockAttentionHead()}

                    # Test cases
                    method = exec_locals['_get_input_space_from_node']
                    self.assertEqual(method(None, node_mlp), "mlp_input_space",
                                     f"Code snippet {i} failed to return correct input space for MLP.")
                    self.assertEqual(method(None, node_attention), "left_space,right_space,attention_input_space",
                                     f"Code snippet {i} failed to return correct input space for AttentionHead.")
                    self.assertIsNone(method(None, {"model_block": object()}),
                                     f"Code snippet {i} returned non-None for unrecognized block.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_get_input_space_from_node",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_get_input_space_from_node",
                        "code": code,
                        "result": "failed"
                    })

        # Test Summary
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

        # Remove old records for _get_input_space_from_node
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_get_input_space_from_node"
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