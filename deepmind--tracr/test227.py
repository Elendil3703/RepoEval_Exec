import unittest
import json
import os
from typing import Iterable, List, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class AttentionHead:
    pass

class MultiAttentionHead:
    def heads(self) -> Iterable[AttentionHead]:
        return iter([])

class TestHeadsMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[226]  # Get the 227th JSON element (index 226)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet at index 226.")

    def test_heads_function(self):
        """Dynamically test the heads function behavior."""
        code = self.code_snippet
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        exec_globals = {
            'AttentionHead': AttentionHead,
            'MultiAttentionHead': MultiAttentionHead,
            'Iterable': Iterable,
        }
        exec_locals = {}

        try:
            # Execute the provided code
            exec(code, exec_globals, exec_locals)

            # Check if 'heads' function exists in the executed context
            if 'heads' not in exec_locals:
                print("Code snippet: FAILED, 'heads' method not found.\n")
                failed_count += 1
                results.append({
                    "function_name": "heads",
                    "code": code,
                    "result": "failed"
                })
                return

            # Create a class with a heads method
            sub_blocks = [
                AttentionHead(),
                MultiAttentionHead(),  # Let it return empty for now
                "invalid"  # This should trigger the NotImplementedError
            ]

            class TestClass:
                def __init__(self, sub_blocks):
                    self.sub_blocks = sub_blocks

                heads = exec_locals['heads']

            instance = TestClass(sub_blocks)

            # Test: heads should return an iterable of AttentionHead
            heads_list = list(instance.heads())
            self.assertTrue(
                all(isinstance(h, AttentionHead) for h in heads_list),
                "The 'heads' method did not return only AttentionHead instances."
            )

            passed_count += 1
            results.append({
                "function_name": "heads",
                "code": code,
                "result": "passed"
            })

        except NotImplementedError:
            print("Code snippet: FAILED with a NotImplementedError.\n")
            failed_count += 1
            results.append({
                "function_name": "heads",
                "code": code,
                "result": "failed"
            })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "heads",
                "code": code,
                "result": "failed"
            })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= Write Results to test_result.jsonl =============
        # Read existing test_result.jsonl (if not present, ignore)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for heads function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "heads"
        ]

        # Append new results
        existing_records.extend(results)

        # Overwrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()