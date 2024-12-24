import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestUnembedResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[185]  # Get the 186th JSON element (index 185)
        if not cls.code_snippet:
            raise ValueError("The selected JSON element contains no code snippets.")

    def test_unembed_function(self):
        """Dynamically test the unembed function in the JSON snippet."""
        results = []
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests

        # Extract the specific code snippet
        code = self.code_snippet

        # Static checks
        if "def unembed" not in code:
            print("Code snippet: FAILED, 'unembed' function definition not found.\n")
            failed_count += 1
            results.append({
                "function_name": "unembed",
                "code": code,
                "result": "failed"
            })
        else:
            exec_globals = {
                'Any': Any,  # Ensure Any is available
                'assemble': assemble_mock(),  # Inject a mock of the assemble module
            }
            exec_locals = {}
            function_verified = False

            try:
                # Execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Check if the unembed function exists after execution
                unembed_func = exec_locals.get('unembed')
                if not unembed_func:
                    raise Exception("unembed function not found in exec_locals.")

                # Attempt to call the unembed function with a test embedding
                test_embeddings = [0.1, 0.2, 0.3]
                result = unembed_func(test_embeddings)

                # Verify the returned result
                self.assertIsNotNone(
                    result,
                    "Unembed function did not return a result."
                )

                function_verified = True
                print("Code snippet: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "unembed",
                    "code": code,
                    "result": "passed"
                })

            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                results.append({
                    "function_name": "unembed",
                    "code": code,
                    "result": "failed"
                })
                failed_count += 1
            
            # Ensure the function was verified
            self.assertTrue(function_verified, "Unembed function was not correctly verified.")
        
        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        
        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the unembed function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "unembed"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

def assemble_mock():
    class MockAssemble:
        def _make_embedding_modules(self, **kwargs):
            return self

        def unembed(self, embeddings, use_unembed_argmax):
            # Mock implementation of the unembed method
            if use_unembed_argmax:
                return [embed * 2 for embed in embeddings]
            else:
                return embeddings
    
    return MockAssemble()

if __name__ == "__main__":
    unittest.main()