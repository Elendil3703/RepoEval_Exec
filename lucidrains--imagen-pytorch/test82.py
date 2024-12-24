import unittest
import json
import os
import torch.nn as nn

TEST_RESULT_JSONL = "test_result.jsonl"

class RepoEvalCode82Result(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[81]  # Get the 82nd JSON element (0-based index)

        if not cls.code_snippet.strip():
            raise ValueError("The code snippet for test is empty!")

    def test_initialize_layers(self):
        """Test the __init__ function to ensure layers are initialized correctly."""
        exec_globals = {
            'nn': nn
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Define a dummy Attention class for testing initialization
            class Attention(nn.Module):
                def __init__(self, dim, heads, dim_head, context_dim):
                    super().__init__()

            # Define a dummy FeedForward class for testing initialization
            class FeedForward(nn.Module):
                def __init__(self, dim, mult=2):
                    super().__init__()

            # Inject the classes
            exec_globals['Attention'] = Attention
            exec_globals['FeedForward'] = FeedForward

            # Perform the initialization test
            test_instance = exec_locals['SomeClassName'](dim=64, depth=3)
            self.assertEqual(len(test_instance.layers), 3, "Incorrect number of layers initialized.")
            for layer in test_instance.layers:
                self.assertIsInstance(layer, nn.ModuleList, "Layer is not an instance of nn.ModuleList.")
                self.assertIsInstance(layer[0], Attention, "Attention module not initialized correctly.")
                self.assertIsInstance(layer[1], FeedForward, "FeedForward module not initialized correctly.")
            
            result = {
                "function_name": "__init__",
                "code": self.code_snippet,
                "result": "passed"
            }
        except Exception as e:
            result = {
                "function_name": "__init__",
                "code": self.code_snippet,
                "result": "failed",
                "error": str(e)
            }

        # Write result to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove existing records with the function_name "__init__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # Append the new result
        existing_records.append(result)

        # Rewrite the test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()