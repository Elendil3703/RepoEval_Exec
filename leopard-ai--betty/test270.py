import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class DummyClass:
    """A dummy class to test configure_distributed_training."""
    def __init__(self):
        self._strategy = None
        self._backend = None
        self._world_size = None
        self._rank = None
        self._local_rank = None

    def configure_distributed_training(self, dictionary):
        """
        Set the configuration for distributed training.

        :param dictionary: Python dictionary of distributed training provided by Engine.
        :type dictionary: dict
        """
        self._strategy = dictionary["strategy"]
        self._backend = dictionary["backend"]
        self._world_size = dictionary["world_size"]
        self._rank = dictionary["rank"]
        self._local_rank = dictionary["local_rank"]

class TestConfigureDistributedTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[269]  # Get the element with index 269
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_configure_distributed_training(self):
        """Dynamically test the configure_distributed_training function in the JSON."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                exec_globals = {}
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if `configure_distributed_training` is present
                    if 'configure_distributed_training' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'configure_distributed_training' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "configure_distributed_training",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create an instance of DummyClass and use the configure_distributed_training method
                    dummy_instance = DummyClass()
                    config_dict = {
                        "strategy": "ddp",
                        "backend": "nccl",
                        "world_size": 4,
                        "rank": 2,
                        "local_rank": 1
                    }
                    dummy_instance.configure_distributed_training(config_dict)

                    # Verify that the configuration was set correctly
                    self.assertEqual(dummy_instance._strategy, config_dict["strategy"])
                    self.assertEqual(dummy_instance._backend, config_dict["backend"])
                    self.assertEqual(dummy_instance._world_size, config_dict["world_size"])
                    self.assertEqual(dummy_instance._rank, config_dict["rank"])
                    self.assertEqual(dummy_instance._local_rank, config_dict["local_rank"])

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "configure_distributed_training",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "configure_distributed_training",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "configure_distributed_training"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "configure_distributed_training"
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