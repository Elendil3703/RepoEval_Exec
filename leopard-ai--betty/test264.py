import unittest
import json
import sys
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestStepFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[263]  # Get the 264th JSON element

    def test_step_function(self):
        """Test the step function for expected behavior."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Results to be written to JSONL

        for i, code in enumerate(self.code_snippet):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Check for `step` method presence
                if "def step(" not in code:
                    print(f"Code snippet {i}: FAILED, 'step' method not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "step",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {}
                exec_locals = {}

                try:
                    # Execute the snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if a class containing 'step' method is defined
                    step_defined = any(
                        hasattr(cls, 'step') 
                        for cls in exec_locals.values() 
                        if isinstance(cls, type)
                    )

                    if not step_defined:
                        print(f"Code snippet {i}: FAILED, 'step' method not correctly defined.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "step",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # For this test, we assume there's a class that defines the 'step' method
                    for cls_name, cls in exec_locals.items():
                        if isinstance(cls, type) and hasattr(cls, 'step'):
                            instance = cls()
                            # Mock attributes and ensure the `step` logic executes without errors
                            setattr(instance, '_count', 10)
                            setattr(instance, '_unroll_steps', 5)
                            setattr(instance, 'gas', 2)
                            setattr(instance, 'warmup_steps', 5)
                            instance.step_normal = lambda global_step: None
                            instance.step_after_roll_back = lambda: None

                            # Call the step function with a sample global_step
                            instance.step(global_step=100)
                             
                            print(f"Code snippet {i}: PASSED all assertions.\n")
                            passed_count += 1
                            results.append({
                                "function_name": "step",
                                "code": code,
                                "result": "passed"
                            })
                            break

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "step",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippet)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippet), "Test count mismatch!")

        # Writing results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove previous "step" function records
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "step"]

        # Append new results
        existing_records.extend(results)

        # Rewrite the JSONL
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()