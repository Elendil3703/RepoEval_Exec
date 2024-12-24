import unittest
import json
import sys
import os
import torch
from torch.utils.data import DataLoader, random_split
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class SampleData:
    """A mock dataset class for testing purposes."""
    def __len__(self):
        return 100

def exists(obj):
    """Utility function to check if an object exists (is not None)."""
    return obj is not None

class Model:
    """A mock model class to be tested."""
    def __init__(self, split_valid_from_train=False, split_valid_fraction=0.2, split_random_seed=42):
        self.train_dl = None
        self.valid_dl = None
        self.split_valid_from_train = split_valid_from_train
        self.split_valid_fraction = split_valid_fraction
        self.split_random_seed = split_random_seed

    def print(self, msg):
        pass

    def add_train_dataloader(self, dl):
        self.train_dl = dl

    def add_valid_dataset(self, ds, batch_size, **dl_kwargs):
        self.valid_dl = DataLoader(ds, batch_size=batch_size, **dl_kwargs)

    def add_train_dataset(self, ds=None, *, batch_size, **dl_kwargs):
        if not exists(ds):
            return

        assert not exists(self.train_dl), 'training dataloader was already added'

        valid_ds = None
        if self.split_valid_from_train:
            train_size = int((1 - self.split_valid_fraction) * len(ds))
            valid_size = len(ds) - train_size

            ds, valid_ds = random_split(ds, [train_size, valid_size], generator=torch.Generator().manual_seed(self.split_random_seed))
            self.print(f'training with dataset of {len(ds)} samples and validating with randomly splitted {len(valid_ds)} samples')

        dl = DataLoader(ds, batch_size=batch_size, **dl_kwargs)
        self.add_train_dataloader(dl)

        if not self.split_valid_from_train:
            return

        self.add_valid_dataset(valid_ds, batch_size=batch_size, **dl_kwargs)

class TestModelTrainDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[107]  # Get the 108th JSON element

    def test_add_train_dataset(self):
        """Test addition of training datasets with various configurations."""
        results = []  # Collect results for appending to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Dynamic execution environment
                exec_globals = {
                    'torch': torch,
                    'random_split': random_split,
                    'DataLoader': DataLoader,
                    'exists': exists,
                    'SampleData': SampleData
                }
                exec_locals = {}

                try:
                    # Ensure code can be executed correctly
                    exec(code, exec_globals, exec_locals)

                    # Ensure add_train_dataset function exists in executed code
                    if 'add_train_dataset' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, function 'add_train_dataset' not found.\n")
                        results.append({
                            "function_name": "add_train_dataset",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    add_train_dataset = exec_locals['add_train_dataset']
                    model = Model(split_valid_from_train=True)

                    # Test adding train dataset
                    ds = SampleData()
                    model.add_train_dataset(ds=ds, batch_size=10)
                    
                    self.assertIsNotNone(model.train_dl, f"Train dataloader not created in snippet {i}.")
                    self.assertEqual(len(model.train_dl.dataset), 80, f"Incorrect train dataset size in snippet {i}.")
                    
                    # Test valid dataset creation
                    self.assertIsNotNone(model.valid_dl, f"Validation dataloader not created in snippet {i}.")
                    self.assertEqual(len(model.valid_dl.dataset), 20, f"Incorrect validation dataset size in snippet {i}.")
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    results.append({
                        "function_name": "add_train_dataset",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    results.append({
                        "function_name": "add_train_dataset",
                        "code": code,
                        "result": "failed"
                    })

        # Write the results into test_result.jsonl
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
            if rec.get("function_name") != "add_train_dataset"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()