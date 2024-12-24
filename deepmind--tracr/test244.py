import unittest
import json
import os
from typing import Any, NamedTuple  # 确保注入的环境中有 Any, NamedTuple
import jax.numpy as jnp

TEST_RESULT_JSONL = "test_result.jsonl"

class CompiledTransformerModelOutput(NamedTuple):
    transformer_output: Any
    unembedded_output: Any

class TestCallMethod(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[243]  # Get the 244th JSON element (index 243)

    def test_call_method(self):
        """Dynamically test __call__ method in the JSON with additional checks."""
        passed_count = 0  
        failed_count = 0  
        results = []      

        code = self.code_snippet

        # Inject custom global variables
        exec_globals = {
            'Any': Any,
            'CompiledTransformerModelOutput': CompiledTransformerModelOutput,
            'jnp': jnp,
        }
        exec_locals = {}

        try:
            # Dynamically execute code snippet
            exec(code, exec_globals, exec_locals)

            # Retrieve call method
            if '__call__' not in exec_locals:
                raise ValueError("'__call__' method not found in executed locals.")

            call_method = exec_locals['__call__']

            # Test inputs
            tokens = jnp.array([1, 2, 3, 4])
            exec_globals['self'] = type('MockModel', (object,), {
                'pad_token': None,
                'embed': lambda self, x: x + 1,  # Mock embedding function
                'transformer': lambda self, x, mask, use_dropout: type('TransformOutput', (object,), {'output': x * mask}),
                'unembed': lambda self, x, use_unembed_argmax: x.sum(),
                'use_unembed_argmax': True
            })()

            # Call the method
            output = call_method(exec_globals['self'], tokens)

            # Assertions
            self.assertIsInstance(output, CompiledTransformerModelOutput, "Output is not of type CompiledTransformerModelOutput.")
            self.assertEqual(output.transformer_output.output.tolist(), [2, 3, 4, 5], "Transformer output is not as expected.")
            self.assertEqual(output.unembedded_output, 14, "Unembedded output is not as expected.")

            print("Code snippet: PASSED all assertions.")
            passed_count += 1
            results.append({
                "function_name": "__call__",
                "code": code,
                "result": "passed"
            })
            
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "__call__",
                "code": code,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write the test results into test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for __call__
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__call__"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()