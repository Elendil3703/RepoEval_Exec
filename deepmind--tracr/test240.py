import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxForwardResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[239]  # Get the 240th JSON element (index 239)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the selected JSON array")

    def test_code_snippets(self):
        """Test forward logic in the code snippets."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                if "def forward" not in code:
                    print(f"Code snippet {i}: FAILED, 'forward' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Prepare execution environment
                exec_globals = {
                    'compressed_model': mock_compressed_model,
                    'model': mock_model,
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Verify the function 'forward'
                    self.assertIn(
                        'forward',
                        exec_locals,
                        f"Code snippet {i} did not define 'forward'."
                    )
                    
                    forward_func = exec_locals['forward']
                    
                    # Mock inputs
                    emb = "mock_emb"
                    mask = "mock_mask"
                    
                    # Call the forward function
                    result = forward_func(emb, mask)

                    # Check if the result is correct (specific checks depend on the mocked behavior)
                    self.assertEqual(result, "expected_result", f"Code snippet {i} did not return the expected result.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write to test_result.jsonl
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
            if rec.get("function_name") != "forward"
        ]
        
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

def mock_compressed_model():
    class CompressedTransformer:
        def __init__(self, config):
            pass

        def __call__(self, emb, mask, embedding_size=None, unembed_at_every_layer=None):
            return "expected_result"

    return CompressedTransformer

def mock_model():
    class TransformerConfig:
        def __init__(self, num_heads, num_layers, key_size, mlp_hidden_size, dropout_rate, causal, layer_norm):
            pass

    return TransformerConfig

if __name__ == "__main__":
    unittest.main()