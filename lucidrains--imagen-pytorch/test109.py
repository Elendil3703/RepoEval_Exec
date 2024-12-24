import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class MockDataLoader:
    def __iter__(self):
        return iter(range(100))

def exists(obj):
    return obj is not None

class MockTrainer:
    def __init__(self):
        self.train_dl = None
        self.train_dl_iter = None

    def cycle(self, data_loader):
        while True:
            for data in data_loader:
                yield data

    def create_train_iter(self):
        assert exists(self.train_dl), 'training dataloader has not been registered with the trainer yet'
        
        if exists(self.train_dl_iter):
            return
        
        self.train_dl_iter = self.cycle(self.train_dl)

class TestCreateTrainIter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[108]  # Get the 109th code snippet (index 108)

    def test_create_train_iter(self):
        """Test the create_train_iter logic from the provided code snippet."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet

        exec_globals = {
            'MockDataLoader': MockDataLoader,
            'exists': exists,
            'Any': Any,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Instantiate a mock trainer
            trainer = MockTrainer()

            # Check assertion failure when train_dl is not set
            try:
                trainer.create_train_iter()
                self.fail("ValueError expected when train_dl is unset")
            except AssertionError as e:
                self.assertEqual(str(e), 'training dataloader has not been registered with the trainer yet')

            # Set train_dl and test the creation of train_dl_iter
            trainer.train_dl = MockDataLoader()

            # Ensure that train_dl_iter is not set initially
            self.assertIsNone(trainer.train_dl_iter)

            # Call create_train_iter to set train_dl_iter
            trainer.create_train_iter()

            # Cycle must be called from train_dl
            self.assertIsNotNone(trainer.train_dl_iter)

            for _ in range(5):  # Test the cycling
                next_item = next(trainer.train_dl_iter)
                self.assertIn(next_item, range(100))

            print(f"Code snippet 'create_train_iter': PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "create_train_iter",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet 'create_train_iter': FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "create_train_iter",
                "code": code,
                "result": "failed"
            })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Writing results to test_result.jsonl
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
            if rec.get("function_name") != "create_train_iter"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()