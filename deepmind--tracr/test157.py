import unittest
import json
import os
from typing import Any  # Ensure Any is available in the testing environment

TEST_RESULT_JSONL = "test_result.jsonl"
GRAPH_STRUCTURE = {
    "edges": [],
    "nodes": {}
}

class MockRasp:
    """Mock class to simulate the rasp.SOp."""
    pass

class MockNode:
    """Mock class to simulate nodes with different attributes."""
    def __init__(self, expr, output_basis):
        self.expr = expr
        self.output_basis = output_basis

class MockBasis:
    """Mock class for bases.VectorSpaceWithBasis"""
    def __init__(self, basis):
        self.basis = basis

    def issubspace(self, other):
        # For testing purposes, assume all spaces are subspaces
        return True

def get_input_space_from_node(node):
    """Mock function to get input space from a node."""
    return MockBasis(node.output_basis)

class TestCheckSpacesAreConsistent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[156]  # Get the 157th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 157th JSON array")

    def test_check_spaces_are_consistent(self):
        """Dynamically test the code related to _check_spaces_are_consistent in the JSON."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for _check_spaces_are_consistent snippet {i}...")

                # ------------------- Static Checks -------------------
                if "_check_spaces_are_consistent" not in code:
                    print(f"Code snippet {i}: FAILED, '_check_spaces_are_consistent' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_check_spaces_are_consistent",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'MockRasp': MockRasp,
                    'MockNode': MockNode,
                    'MockBasis': MockBasis,
                    'nodes': {
                        'EXPR': 'expr',
                        'OUTPUT_BASIS': 'output_basis'
                    },
                    'rasp': {
                        'SOp': MockRasp
                    },
                    'bases': {
                        'VectorSpaceWithBasis': MockBasis
                    },
                    '_check_spaces_are_consistent': None,  # Placeholder
                    'get_input_space_from_node': get_input_space_from_node
                }

                exec_locals = {}

                try:
                    # Dynamic execution of the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure the function was executed and implemented correctly
                    if '_check_spaces_are_consistent' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_check_spaces_are_consistent' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_check_spaces_are_consistent",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create a mock graph
                    graph = {
                        "edges": [(1, 2)],
                        "nodes": {
                            1: MockNode(MockRasp(), MockBasis(['basis1'])),
                            2: MockNode(MockRasp(), MockBasis(['basis2']))
                        }
                    }

                    # Test the function with a mock graph object
                    test_instance = self
                    exec_locals['_check_spaces_are_consistent'](test_instance, graph)
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_check_spaces_are_consistent",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_check_spaces_are_consistent",
                        "code": code,
                        "result": "failed"
                    })

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for _check_spaces_are_consistent
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_check_spaces_are_consistent"
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