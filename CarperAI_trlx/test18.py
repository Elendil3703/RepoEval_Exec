import unittest
import json
import os
from unittest.mock import MagicMock
from torch import nn

TEST_RESULT_JSONL = "test_result.jsonl"

class TestHfGetDecoderFinalNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[17]  # Get the 18th JSON element (index 17)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_hf_get_decoder_final_norm(self):
        """Test the hf_get_decoder_final_norm logic with mock objects."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                exec_globals = {
                    'nn': nn,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if hf_get_decoder_final_norm exists
                    if 'hf_get_decoder_final_norm' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'hf_get_decoder_final_norm' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "hf_get_decoder_final_norm",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    hf_get_decoder_final_norm = exec_locals['hf_get_decoder_final_norm']

                    # Mock models with expected norm attributes
                    model_mock_1 = MagicMock()
                    setattr(model_mock_1, "transformer", MagicMock(ln_f=nn.LayerNorm(10)))
                    findattr = lambda model, attrs: getattr(model.transformer, 'ln_f', None)

                    model_mock_2 = MagicMock()
                    setattr(model_mock_2, "model", MagicMock(decoder=MagicMock(final_layer_norm=nn.LayerNorm(10))))
                    findattr = lambda model, attrs: getattr(model.model.decoder, 'final_layer_norm', None)

                    norm_1 = hf_get_decoder_final_norm(model_mock_1)
                    norm_2 = hf_get_decoder_final_norm(model_mock_2)

                    self.assertIsInstance(norm_1, nn.LayerNorm, "hf_get_decoder_final_norm should return a LayerNorm instance.")
                    self.assertIsInstance(norm_2, nn.LayerNorm, "hf_get_decoder_final_norm should return a LayerNorm instance.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "hf_get_decoder_final_norm",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "hf_get_decoder_final_norm",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary info
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the test results to test_result.jsonl
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
            if rec.get("function_name") != "hf_get_decoder_final_norm"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()