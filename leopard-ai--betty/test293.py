import unittest
import json
import os
import torch
import torch.nn.functional as F

TEST_RESULT_JSONL = "test_result.jsonl"

class DummyInnerModel:
    def __call__(self, inputs):
        # 模拟模型输出，实际上会根据模型做特定的操作
        return torch.sigmoid(inputs)


class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[292]  # Get the 293rd JSON element (index 292)

    def test_training_step(self):
        """Test the extracted training_step function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect results for JSONL writing

        # Define a batch input
        inputs = torch.tensor([[0.5, -0.5], [1.0, -1.0]], dtype=torch.float)
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float)

        # Expectation using the dummy model
        expected_loss = F.binary_cross_entropy_with_logits(
            DummyInnerModel()(inputs),
            targets
        )

        # =============
        # Execute the code snippet dynamically
        # =============
        exec_globals = {
            'torch': torch,
            'F': F,
            'DummyInnerModel': DummyInnerModel,
        }
        exec_locals = {}

        try:
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check if training_step is defined
            if 'training_step' not in exec_locals:
                print("FAIL: 'training_step' not found in exec_locals.")
                failed_count += 1
                results.append({
                    "function_name": "training_step",
                    "code": self.code_snippet,
                    "result": "failed"
                })
            else:
                # Prepare a dummy object with the appropriate method
                class DummyObject:
                    def __init__(self, inner):
                        self.inner = inner

                dummy_instance = DummyObject(DummyInnerModel())
                loss = exec_locals['training_step'](dummy_instance, (inputs, targets))

                # Check that loss calculation is as expected
                if torch.isclose(loss, expected_loss):
                    print("PASS: Correct loss calculated.")
                    passed_count += 1
                    results.append({
                        "function_name": "training_step",
                        "code": self.code_snippet,
                        "result": "passed"
                    })
                else:
                    print("FAIL: Incorrect loss value.")
                    failed_count += 1
                    results.append({
                        "function_name": "training_step",
                        "code": self.code_snippet,
                        "result": "failed"
                    })

        except Exception as e:
            print(f"FAIL: Exception during execution: {e}")
            failed_count += 1
            results.append({
                "function_name": "training_step",
                "code": self.code_snippet,
                "result": "failed"
            })

        # =========== Write results to test_result.jsonl ===========
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
            if rec.get("function_name") != "training_step"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()