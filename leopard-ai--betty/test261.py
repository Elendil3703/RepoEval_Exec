import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPatchOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[260]  # Get the 261st JSON element

    def test_patch_optimizer(self):
        """Test patch_optimizer function for correct behavior."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        code = self.code_snippet
        # Simulated context for exec
        class MockOptimizer: pass

        exec_globals = {
            'patch_optimizer': None,
            'MockOptimizer': MockOptimizer,
        }

        exec_locals = {}

        try:
            # Dynamically execute code snippet
            exec(code, exec_globals, exec_locals)

            patch_optimizer_func = exec_locals.get('patch_optimizer')
            if not callable(patch_optimizer_func):
                raise Exception("'patch_optimizer' not found or is not callable")

            class TestSystem:
                def __init__(self, strategy):
                    self._strategy = strategy
                    self.optimizer = MockOptimizer()
                
                def trainable_parameters(self):
                    return ['param1', 'param2']

                def param_groups(self):
                    return [{'params': ['param1']}, {'params': ['param2']}]

                def is_implemented(self, method_name):
                    return hasattr(self, method_name)

                def accelerator(self, optimizer):
                    class AcceleratedOptimizer:
                        pass
                    return AcceleratedOptimizer()

            # Test cases
            strategies = ['ddp', 'fsdp', 'zero', 'accelerate']
            expected_classes = [MockOptimizer, MockOptimizer, MockOptimizer, object]  # Simulate result types

            for strategy, expected_class in zip(strategies, expected_classes):
                system = TestSystem(strategy)
                patch_optimizer_func(system)

                self.assertIsInstance(
                    system.optimizer,
                    expected_class,
                    f"Optimizer not correctly patched for strategy: {strategy}",
                )
                passed_count += 1

                results.append({
                    "function_name": "patch_optimizer",
                    "strategy": strategy,
                    "result": "passed"
                })

            print(f"All tests passed for code snippet.")

        except Exception as e:
            print(f"Code snippet test failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "patch_optimizer",
                "code": code,
                "result": "failed"
            })

        # Final statistics
        self.assertEqual(passed_count + failed_count, len(strategies), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "patch_optimizer"
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