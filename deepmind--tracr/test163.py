import unittest
import json
import re
import os
from typing import Sequence, Any

TEST_RESULT_JSONL = "test_result.jsonl"


class Node:
    """Mock Node class for testing purposes."""
    def __init__(self, layer_type: str):
        self.layer_type = layer_type


def _node_is_mlp(node: Node) -> bool:
    """Mock check function for determining if a node is an MLP node."""
    return node.layer_type == "mlp"


class TestNodeMLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[162]  # Get the 163rd element (index 162)
        if not cls.code_snippet:
            raise ValueError("Expected the code snippet to not be empty.")

    def test_all_mlp_nodes(self):
        """Test _all_mlp_nodes with various node list cases."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        
        # ------------------- Static checks -------------------
        if "_all_mlp_nodes" not in code:
            print("FAILED, '_all_mlp_nodes' not found in code.")
            failed_count += 1
            results.append({
                "function_name": "_all_mlp_nodes",
                "code": code,
                "result": "failed"
            })
            return

        func_pattern = r"def\s+_all_mlp_nodes\s*\("
        if not re.search(func_pattern, code):
            print("FAILED, incorrect signature for '_all_mlp_nodes'.")
            failed_count += 1
            results.append({
                "function_name": "_all_mlp_nodes",
                "code": code,
                "result": "failed"
            })
            return

        # ------------------- Dynamic execution and logic tests -------------------
        exec_globals = {'Node': Node, '_node_is_mlp': _node_is_mlp, 'Sequence': Sequence, 'Any': Any}
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(code, exec_globals, exec_locals)

            if '_all_mlp_nodes' not in exec_locals:
                print("FAILED, '_all_mlp_nodes' not found in exec_locals.")
                failed_count += 1
                results.append({
                    "function_name": "_all_mlp_nodes",
                    "code": code,
                    "result": "failed"
                })
                return

            # Test scenarios
            test_cases = [
                ([], True),
                ([Node('mlp')], True),
                ([Node('mlp'), Node('mlp')], True),
                ([Node('mlp'), Node('non_mlp')], False),
                ([Node('non_mlp')], False),
            ]

            for i, (nodes, expected) in enumerate(test_cases):
                with self.subTest(test_case=i):
                    result = exec_locals['_all_mlp_nodes'](nodes)
                    self.assertEqual(result, expected, f"Test case {i} failed.")
            
            print("All test cases passed.")
            passed_count += len(test_cases)
            results.append({
                "function_name": "_all_mlp_nodes",
                "code": code,
                "result": "passed"
            })
        
        except Exception as e:
            print(f"FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "_all_mlp_nodes",
                "code": code,
                "result": "failed"
            })

        # ======= Write results to test_result.jsonl =======
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
            if rec.get("function_name") != "_all_mlp_nodes"
        ]

        existing_records.extend(results)

        # Write updated results
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()