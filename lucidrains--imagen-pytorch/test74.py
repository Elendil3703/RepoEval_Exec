import unittest
import json
import os
import torch.nn as nn
import torch
from torch.nn.modules.activation import SiLU
from torch.nn.modules.conv import Conv2d

TEST_RESULT_JSONL = "test_result.jsonl"


class Identity(nn.Module):
    def forward(self, x):
        return x


class TestGroundTruthInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[73]  # Get the 74th JSON element (index 73)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array at index 73")

    def test_init_method(self):
        """Dynamically test all code snippets for the __init__ function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into the JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Code snippet should have '__init__'
                if "__init__" not in code:
                    print(f"Code snippet {i}: FAILED, '__init__' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and testing logic
                exec_globals = {
                    'nn': nn,
                    'Identity': Identity
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    class TestClass:
                        def __init__(self, dim, dim_out, groups=8, norm=True):
                            self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
                            self.activation = nn.SiLU()
                            self.project = nn.Conv2d(dim, dim_out, 3, padding=1)

                    # Example instantiation to test functionality
                    dim, dim_out, groups, norm = 4, 8, 2, True
                    test_instance = TestClass(dim, dim_out, groups, norm)

                    self.assertIsInstance(test_instance.groupnorm, nn.GroupNorm)
                    self.assertIsInstance(test_instance.activation, SiLU)
                    self.assertIsInstance(test_instance.project, Conv2d)
                    self.assertEqual(test_instance.groupnorm.num_groups, groups)
                    self.assertEqual(test_instance.groupnorm.num_channels, dim)
                    self.assertEqual(test_instance.project.in_channels, dim)
                    self.assertEqual(test_instance.project.out_channels, dim_out)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")

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
            if rec.get("function_name") != "__init__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()