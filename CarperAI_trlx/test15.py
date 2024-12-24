import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestRHasAttr(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[14]  # Get the 75th JSON element (0-indexed)
        if not cls.code_snippet:
            raise ValueError("Expected a valid code snippet in the JSON data")

    def test_rhasattr(self):
        """Test the rhasattr function dynamically."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        code = self.code_snippet
        print("Testing the rhasattr function...")

        # ------------------- Dynamic Execution -------------------
        exec_globals = {}
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if rhasattr is defined
            if 'rhasattr' not in exec_locals:
                print("FAILED: 'rhasattr' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "rhasattr",
                    "code": code,
                    "result": "failed"
                })
                return

            rhasattr = exec_locals['rhasattr']

            # Test 1: Single-level attribute
            class Obj1:
                attr1 = 10

            self.assertTrue(
                rhasattr(Obj1, "attr1"),
                "rhasattr should return True for an existing single-level attribute"
            )

            # Test 2: Multi-level attribute
            class Obj2:
                class Nested:
                    attr2 = 20

                nested = Nested()

            self.assertTrue(
                rhasattr(Obj2, "nested.attr2"),
                "rhasattr should return True for an existing multi-level attribute"
            )

            # Test 3: Non-existing attribute
            self.assertFalse(
                rhasattr(Obj2, "nested.nonexistent"),
                "rhasattr should return False for a non-existing attribute"
            )

            # Test 4: Incorrect chain
            self.assertFalse(
                rhasattr(Obj2, "nested.nonexistent.attr"),
                "rhasattr should return False for an invalid attribute chain"
            )

            # Test 5: Empty attribute string
            self.assertFalse(
                rhasattr(Obj2, ""),
                "rhasattr should return False for an empty attribute string"
            )

            print("PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "rhasattr",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "rhasattr",
                "code": code,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= Write Test Results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for rhasattr
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "rhasattr"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()