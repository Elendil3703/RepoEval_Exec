import unittest
import json
import os
from typing import Sequence
import networkx as nx

# Constants
TEST_RESULT_JSONL = "test_result.jsonl"

# Placeholder Node and NodeID classes for testing
class Node:
    def __init__(self, id):
        self.id = id

class NodeID:
    pass

# Simulated enum or constant used in the node attribute
class nodes:
    ID = 'id'
    EXPR = 'expr'

# Placeholder class for rasp.SOp
class SOp:
    pass

# Function to test
def _get_longest_path_length_to_node(graph: nx.DiGraph, sources: Sequence[Node],
                                     node: Node) -> int:
    """Returns the lengths of the longest path from sources to node.
    Only SOps count towards the length of a path.
    Args:
        graph: DAG to compute longest path in.
        sources: List of starting nodes, longest path will be a maximum over all.
        node: Target node.

    Returns:
        Number of steps needed for the longest path from the source to the node, or
        -1 if there is no path from any of the sources to the target node.
    """
    if node in sources:
        return 0

    def num_sops(path: Sequence[NodeID]) -> int:
        num = 0
        for node_id in path:
            if isinstance(graph.nodes[node_id][nodes.EXPR], SOp):
                num += 1
        return num

    result = -1
    for source in sources:
        all_paths = nx.all_simple_paths(graph, source[nodes.ID], node[nodes.ID])
        longest_path_len = max(map(num_sops, all_paths), default=-1) - 1
        if longest_path_len > result:
            result = longest_path_len
    return result


class TestGetLongestPathLengthToNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        # Access the 158th element (index 157)
        cls.test_data = data[157]

    def test_longest_path(self):
        """Tests for _get_longest_path_length_to_node function."""
        passed_count = 0
        failed_count = 0
        results = []

        # Example setup for test
        G = nx.DiGraph()  # Create a directed graph

        # Add nodes with EXPR attribute
        nodes_with_sop = ['A', 'B', 'C']
        for node in nodes_with_sop:
            G.add_node(node, expr=SOp())

        G.add_node('D', expr=None)  # Add a node without SOp

        # Add edges to simulate paths
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])

        # Define sources and target node
        sources = [Node('A')]
        target_node = Node('D')

        # Test the function
        try:
            result = _get_longest_path_length_to_node(G, sources, target_node)
            self.assertEqual(result, 2, "Longest path calculation is incorrect.")
            passed_count += 1
            results.append({
                "function_name": "_get_longest_path_length_to_node",
                "code": self.test_data,
                "result": "passed"
            })
            print("Test passed.")
        except AssertionError as e:
            failed_count += 1
            results.append({
                "function_name": "_get_longest_path_length_to_node",
                "code": self.test_data,
                "result": "failed"
            })
            print("Test failed: ", e)

        # Test Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")

        # Write results to JSONL
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
            if rec.get("function_name") != "_get_longest_path_length_to_node"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()