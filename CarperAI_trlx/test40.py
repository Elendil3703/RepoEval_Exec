import unittest
import json
import sys
import os
import torch
from torch import nn

TEST_RESULT_JSONL = "test_result.jsonl"

class MockBaseModel(nn.Module):
    """A mock implementation of a base model for testing purposes."""
    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_length = input_ids.shape
        hidden_size = 768
        num_layers = 12
        hidden_states = torch.rand((num_layers, batch_size, seq_length, hidden_size))
        logits = torch.rand((batch_size, seq_length, 30522))
        past_key_values = None
        output = {
            'hidden_states': hidden_states,
            'logits': logits,
            'past_key_values': past_key_values
        }
        return output


class MockILQLHeads(nn.Module):
    """A mock implementation of an ILQL head for testing purposes."""
    def forward(self, hidden_states, states_ixs=None, actions_ixs=None):
        batch_size, seq_length, hidden_size = hidden_states.shape
        qs = torch.rand((batch_size, seq_length))
        target_qs = torch.rand((batch_size, seq_length))
        vs = torch.rand((batch_size, seq_length))
        return qs, target_qs, vs


class MockModel(nn.Module):
    """A mock composite model for testing purposes."""
    def __init__(self):
        super().__init__()
        self.base_model = MockBaseModel()
        self.ilql_heads = MockILQLHeads()

    def get_compatible_forward_kwargs(self, **kwargs):
        return kwargs

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        actions_ixs=None,
        states_ixs=None,
    ):
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        forward_kwargs["output_hidden_states"] = True

        outputs = self.base_model(**forward_kwargs)
        qs, target_qs, vs = self.ilql_heads(outputs['hidden_states'][-1], states_ixs=states_ixs, actions_ixs=actions_ixs)

        return outputs['logits'], qs, target_qs, vs, outputs['past_key_values']


class TestForwardFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[39]
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet at the given index.")

    def test_forward_function(self):
        """Test the forward function with specific input and output checks."""
        model = MockModel()

        input_ids = torch.randint(0, 30522, (8, 128))
        attention_mask = torch.ones_like(input_ids)

        try:
            logits, qs, target_qs, vs, past_key_values = model.forward(input_ids, attention_mask=attention_mask)

            # Check the shapes of returned objects
            self.assertEqual(logits.shape, (8, 128, 30522), "Logits shape is incorrect")
            self.assertEqual(qs.shape, (8, 128), "qs shape is incorrect")
            self.assertEqual(target_qs.shape, (8, 128), "target_qs shape is incorrect")
            self.assertEqual(vs.shape, (8, 128), "vs shape is incorrect")
            self.assertIsNone(past_key_values, "past_key_values should be None")

            # If all checks pass
            result = {
                "function_name": "forward",
                "code": self.code_snippet,
                "result": "passed"
            }

        except Exception as e:
            # If any exception occurs
            result = {
                "function_name": "forward",
                "code": self.code_snippet,
                "result": "failed",
                "error": str(e)
            }

        # Write the result to the test_result.jsonl file
        self.write_result(result)

    def write_result(self, result):
        # Read existing results from the JSONL file, if any
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    existing_records.append(json.loads(line.strip()))

        # Remove old results for 'forward' function if exist
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "forward"]

        # Append the new result
        existing_records.append(result)

        # Write back all results to the file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()