import unittest
import json
import os
from typing import Any  # Included in the execution context

TEST_RESULT_JSONL = "test_result.jsonl"

class DummyTransformer:
    @staticmethod
    def MLP():
        return "MLP"

    @staticmethod
    def AttentionHead():
        return "AttentionHead"

class DummyRasp:
    class SOp: pass
    class Map(SOp): pass
    class SequenceMap(SOp): pass
    class Aggregate(SOp): pass

nodes = {
    'EXPR': 'expr',
    'MODEL_BLOCK': 'model_block'
}

transformers = DummyTransformer
rasp = DummyRasp

class TestCheckBlockTypesAreCorrect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[154]  # Get the 155th JSON element
        if not cls.code_snippet:
            raise ValueError("Expected non-empty code snippet in the 155th JSON element")

    def test_check_block_types_are_correct(self):
        """Test _check_block_types_are_correct function with various graph configurations."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        code = self.code_snippet

        exec_globals = {
            'unittest': unittest,
            'transformers': transformers,
            'rasp': rasp,
            'nodes': nodes
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            if '_check_block_types_are_correct' not in exec_locals:
                raise RuntimeError("_check_block_types_are_correct not defined in the executed code")

            check_function = exec_locals['_check_block_types_are_correct']
            # Define mock graph data
            graphs = [
                (
                    {'node1': {nodes['EXPR']: rasp.Map(), nodes['MODEL_BLOCK']: transformers.MLP()}},
                    True
                ),
                (
                    {'node2': {nodes['EXPR']: rasp.SequenceMap(), nodes['MODEL_BLOCK']: transformers.MLP()}},
                    True
                ),
                (
                    {'node3': {nodes['EXPR']: rasp.Aggregate(), nodes['MODEL_BLOCK']: transformers.AttentionHead()}},
                    True
                ),
                (
                    {'node4': {nodes['EXPR']: rasp.Map(), nodes['MODEL_BLOCK']: transformers.AttentionHead()}},
                    False
                ),
                (
                    {'node5': {nodes['EXPR']: rasp.Aggregate(), nodes['MODEL_BLOCK']: transformers.MLP()}},
                    False
                )
            ]

            for i, (graph, should_pass) in enumerate(graphs):
                with self.subTest(graph_index=i):
                    try:
                        check_function(self, graph)
                        if should_pass:
                            # Passed as expected
                            passed_count += 1
                            results.append({
                                "function_name": "_check_block_types_are_correct",
                                "code": code,
                                "result": "passed"
                            })

                        else:
                            # Failed to catch wrong type
                            print(f"Graph {i}: FAILED, expected a type assertion error.\n")
                            failed_count += 1
                            results.append({
                                "function_name": "_check_block_types_are_correct",
                                "code": code,
                                "result": "failed"
                            })

                    except AssertionError:
                        if not should_pass:
                            # Caught the type assertion error as expected
                            passed_count += 1
                            results.append({
                                "function_name": "_check_block_types_are_correct",
                                "code": code,
                                "result": "passed"
                            })
                        else:
                            # Unexpected error occurred
                            print(f"Graph {i}: FAILED, caught unexpected assertion error.\n")
                            failed_count += 1
                            results.append({
                                "function_name": "_check_block_types_are_correct",
                                "code": code,
                                "result": "failed"
                            })

            print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(graphs)}\n")
            self.assertEqual(passed_count + failed_count, len(graphs), "Test count mismatch!")

        except Exception as e:
            print(f"Execution of code snippet resulted in failure: {e}\n")
            results.append({
                "function_name": "_check_block_types_are_correct",
                "code": code,
                "result": "failed"
            })

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for _check_block_types_are_correct
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_check_block_types_are_correct"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()