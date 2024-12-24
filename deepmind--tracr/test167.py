import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestEnsureNodeFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[166]  # Get the 167th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_ensure_node(self):
        """Dynamically test all code snippets for the 'ensure_node' function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check for presence of 'ensure_node' definition
                if "def ensure_node" not in code:
                    print(f"Code snippet {i}: FAILED, 'ensure_node' function not defined.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "ensure_node",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Prepare global and local execution environments
                exec_globals = {
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    # Check if 'ensure_node' exists after execution
                    if 'ensure_node' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'ensure_node' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "ensure_node",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Setup dummy classes and inputs
                    class MockExpr:
                        def __init__(self, label):
                            self.label = label

                    class MockGraph:
                        def __init__(self):
                            self.nodes = {}

                        def add_node(self, id, **attrs):
                            self.nodes[id] = attrs

                    graph = MockGraph()
                    expr = MockExpr('test_node')
                    
                    # Inject graph into the local context
                    exec_locals['graph'] = graph
                    exec_locals['rasp'] = MockExpr  # Assuming RASPExpr is analogous to MockExpr
                    exec_locals['nodes'] = {"ID": "id", "EXPR": "expression"} 
                    
                    # Test the ensure_node function call
                    node_id = exec_locals['ensure_node'](expr)
                    
                    # Validate the result
                    self.assertEqual(node_id, 'test_node', f"Code snippet {i} did not return expected node_id.")

                    expected_attrs = {'id': 'test_node', 'expression': expr}
                    self.assertIn('test_node', graph.nodes, f"Code snippet {i} did not add node to the graph.")
                    self.assertEqual(graph.nodes['test_node'], expected_attrs,
                                     f"Code snippet {i} did not set correct node attributes in the graph.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "ensure_node",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "ensure_node",
                        "code": code,
                        "result": "failed"
                    })

        # Summary of the test results
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the test results into test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records about "ensure_node" function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "ensure_node"
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