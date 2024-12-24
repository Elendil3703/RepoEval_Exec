import unittest
import json
import os

class TestTokenPosEmbed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[183]  # Get the 184th JSON element (index 183)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_token_pos_embed(self):
        """Dynamically test all code snippets for token_pos_embed function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collecting test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------ Static Checks -------------------
                if "assemble._make_embedding_modules" not in code:
                    print(f"Code snippet {i}: FAILED, '_make_embedding_modules' function not invoked properly.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "token_pos_embed",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------ Dynamic Execution and Test Logic -------------------
                exec_globals = {
                    'assemble': type('mock', (object,), {
                        '_make_embedding_modules': lambda **kwargs: type('MockModules', (object,), {
                            'token_embed': lambda tokens: f"embedded_{tokens}"
                        })()
                    })
                }
                exec_locals = {}

                try:
                    # Dynamic execution of the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check that the function token_pos_embed exists
                    if 'token_pos_embed' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'token_pos_embed' not found after exec.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "token_pos_embed",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the function logic
                    tokens_input = "sample_tokens"
                    expected_output = f"embedded_{tokens_input}"
                    actual_output = exec_locals['token_pos_embed'](tokens_input)

                    self.assertEqual(
                        actual_output,
                        expected_output,
                        f"Code snippet {i} did not produce the expected embedding output."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "token_pos_embed",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "token_pos_embed",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        test_result_jsonl = "test_result.jsonl"
        if os.path.exists(test_result_jsonl):
            with open(test_result_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records with the function_name "token_pos_embed"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "token_pos_embed"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(test_result_jsonl, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()