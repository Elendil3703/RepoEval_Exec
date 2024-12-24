import unittest
import json
import os
import torch
import torch.nn as nn

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_segment = data[64]  # Get the 65th JSON element
        if len(cls.code_segment) < 1:
            raise ValueError("Expected at least one code snippet in the 65th JSON array")

    def test_init_function(self):
        """Dynamically test the __init__ function in the code snippet."""
        results = []  # Collect test results

        code = self.code_segment
        
        # Setup for executing the code
        exec_globals = {
            'torch': torch,
            'nn': nn,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if there's a class with the correct __init__ signature
            defined_class = None
            for obj_name, obj in exec_locals.items():
                if isinstance(obj, type) and '__init__' in obj.__dict__:
                    defined_class = obj
                    break

            self.assertIsNotNone(defined_class, "No class with __init__ found in code.")
            
            # Initialize an instance of the class with provided parameters
            instance = defined_class(dim=128)

            # Follow-up assertions based on the __init__ logic
            self.assertEqual(instance.scale, 8, "__init__ did not set scale correctly.")
            self.assertEqual(instance.heads, 8, "__init__ did not set heads correctly.")
            self.assertIsInstance(instance.norm, nn.LayerNorm, "Expected norm to be a LayerNorm instance.")
            self.assertIsInstance(instance.norm_latents, nn.LayerNorm, "Expected norm_latents to be a LayerNorm instance.")
            self.assertIsInstance(instance.to_q, nn.Linear, "Expected to_q to be a Linear instance.")
            self.assertIsInstance(instance.to_kv, nn.Linear, "Expected to_kv to be a Linear instance.")
            self.assertIsInstance(instance.q_scale, nn.Parameter, "Expected q_scale to be a Parameter instance.")
            self.assertIsInstance(instance.k_scale, nn.Parameter, "Expected k_scale to be a Parameter instance.")
            self.assertIsInstance(instance.to_out, nn.Sequential, "Expected to_out to be a Sequential instance.")

            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "passed"
            })
            print(f"Code snippet with __init__: PASSED all assertions.\n")

        except Exception as e:
            print(f"Code snippet with __init__: FAILED with error: {e}\n")
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for __init__
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
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