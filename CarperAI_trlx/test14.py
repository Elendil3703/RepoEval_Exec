import unittest
import json
import os
import torch
import torch.nn as nn

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMakeHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[13]  # Get the 15th JSON element (0-indexed)
        if not cls.code_snippet:
            raise ValueError("Expected a valid code snippet in the JSON data")

    def test_make_head(self):
        """Test the make_head function dynamically."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        code = self.code_snippet
        print("Testing the make_head function...")

        # ------------------- Dynamic Execution -------------------
        exec_globals = {
            'torch': torch,
            'nn': nn,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if make_head is defined
            if 'make_head' not in exec_locals:
                print("FAILED: 'make_head' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "make_head",
                    "code": code,
                    "result": "failed"
                })
                return

            make_head = exec_locals['make_head']

            # Test 1: Check structure of returned nn.Sequential
            n_embd, out = 32, 10
            dtype = torch.float32
            model = make_head(n_embd, out, dtype)

            self.assertIsInstance(
                model, nn.Sequential,
                "make_head should return an instance of nn.Sequential"
            )
            self.assertEqual(
                len(model), 3,
                "The returned nn.Sequential should contain 3 layers"
            )

            # Test 2: Check layers and types
            self.assertIsInstance(
                model[0], nn.Linear,
                "The first layer should be an instance of nn.Linear"
            )
            self.assertEqual(
                model[0].in_features, n_embd,
                "The first Linear layer's input size is incorrect"
            )
            self.assertEqual(
                model[0].out_features, n_embd * 2,
                "The first Linear layer's output size is incorrect"
            )
            self.assertEqual(
                model[0].weight.dtype, dtype,
                "The first Linear layer's dtype is incorrect"
            )

            self.assertIsInstance(
                model[1], nn.ReLU,
                "The second layer should be an instance of nn.ReLU"
            )

            self.assertIsInstance(
                model[2], nn.Linear,
                "The third layer should be an instance of nn.Linear"
            )
            self.assertEqual(
                model[2].in_features, n_embd * 2,
                "The second Linear layer's input size is incorrect"
            )
            self.assertEqual(
                model[2].out_features, out,
                "The second Linear layer's output size is incorrect"
            )
            self.assertEqual(
                model[2].weight.dtype, dtype,
                "The second Linear layer's dtype is incorrect"
            )

            # Test 3: Check forward pass
            input_tensor = torch.randn(1, n_embd, dtype=dtype)
            output = model(input_tensor)
            self.assertEqual(
                output.shape, (1, out),
                "The output shape of the forward pass is incorrect"
            )

            print("PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "make_head",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "make_head",
                "code": code,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= Write Test Results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for make_head
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "make_head"
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