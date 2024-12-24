import unittest
import json
import sys
import re
import os
import yaml  # Ensure PyYAML is installed
from tempfile import NamedTemporaryFile
from typing import Any  # Ensure Any is available

TEST_RESULT_JSONL = "test_result.jsonl"

class TestLoadYamlFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        if len(data) < 3:
            raise ValueError("Expected at least three code snippets in CarperAI_trlx_result.json")
        cls.code_snippets = data[2]  # Get the third JSON element (index 2)
        # Ensure code_snippets is a list
        if not isinstance(cls.code_snippets, list):
            cls.code_snippets = [cls.code_snippets]
    
    def test_load_yaml_snippets(self):
        """Test the 'load_yaml' function code snippets."""
        results = []
        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Static Checks -------------------
                if "def load_yaml" not in code:
                    print(f"Code snippet {i} FAILED: 'def load_yaml' not found in code snippet.")
                    results.append({
                        "function_name": "load_yaml",
                        "code": code,
                        "result": "failed",
                        "reason": "'def load_yaml' not found in code snippet."
                    })
                    continue  # Skip to next code snippet
                else:
                    func_pattern = r"def\s+load_yaml\s*\(\s*cls\s*,\s*yml_fp\s*:\s*str\s*\)"
                    if not re.search(func_pattern, code):
                        print(f"Code snippet {i} FAILED: Function signature of 'load_yaml' does not match.")
                        results.append({
                            "function_name": "load_yaml",
                            "code": code,
                            "result": "failed",
                            "reason": "Function signature does not match."
                        })
                        continue
                    else:
                        print(f"Code snippet {i}: Passed static checks.")
                        
                # ------------------- Dynamic Execution -------------------
                exec_globals = {}
                exec_locals = {}

                # Mock TRLConfig class
                class TRLConfig:
                    @classmethod
                    def from_dict(cls, config):
                        return cls(config)

                    def __init__(self, config):
                        self.config = config

                exec_globals = {
                    'yaml': yaml,
                    'TRLConfig': TRLConfig,
                    '__name__': '__main__',
                    'Any': Any,
                }

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # After execution, check if 'load_yaml' is defined
                    # Since it's a class method, it might be defined within a class
                    # Let's check if any class defines 'load_yaml' as a method
                    load_yaml_func = None
                    TestClass = None
                    for obj_name, obj in exec_globals.items():
                        if isinstance(obj, type):
                            # Check if 'load_yaml' is a callable attribute of the class
                            if hasattr(obj, 'load_yaml') and callable(getattr(obj, 'load_yaml')):
                                load_yaml_func = getattr(obj, 'load_yaml')
                                TestClass = obj
                                break
                    if load_yaml_func is None:
                        # Try exec_locals
                        for obj_name, obj in exec_locals.items():
                            if isinstance(obj, type):
                                if hasattr(obj, 'load_yaml') and callable(getattr(obj, 'load_yaml')):
                                    load_yaml_func = getattr(obj, 'load_yaml')
                                    TestClass = obj
                                    break
                    if load_yaml_func is None:
                        # If still not found, try to get it directly
                        if 'load_yaml' in exec_globals:
                            load_yaml_func = exec_globals['load_yaml']
                            # Create a mock class to bind the function
                            TestClass = TRLConfig
                        elif 'load_yaml' in exec_locals:
                            load_yaml_func = exec_locals['load_yaml']
                            TestClass = TRLConfig
                        else:
                            print(f"Code snippet {i} FAILED: 'load_yaml' function not found after execution.")
                            results.append({
                                "function_name": "load_yaml",
                                "code": code,
                                "result": "failed",
                                "reason": "'load_yaml' function not found after execution."
                            })
                            continue  # Skip to next code snippet

                        # If load_yaml_func is not bound to a class, bind it
                        if not hasattr(TestClass, 'load_yaml'):
                            setattr(TestClass, 'load_yaml', classmethod(load_yaml_func))

                    # Create temporary YAML file
                    temp_yaml = NamedTemporaryFile(mode='w+', delete=False, suffix='.yaml')
                    temp_yaml.write('param1: value1\nparam2: value2')
                    temp_yaml.close()
                    yml_fp = temp_yaml.name

                    # Call the load_yaml function
                    config_obj = load_yaml_func(TestClass, yml_fp)

                    # Check that config_obj is an instance of TRLConfig
                    self.assertIsInstance(
                        config_obj,
                        TRLConfig,
                        f"Code snippet {i}: Returned object is not an instance of expected class."
                    )
                    # Check that config_obj.config matches the YAML contents
                    expected_config = {
                        'param1': 'value1',
                        'param2': 'value2'
                    }
                    self.assertEqual(
                        config_obj.config,
                        expected_config,
                        f"Code snippet {i}: config_obj.config does not match expected configuration."
                    )
                    os.unlink(temp_yaml.name)  # Remove temporary file

                    # Passed all checks
                    print(f"Code snippet {i} PASSED: Code snippet works as expected.")
                    results.append({
                        "function_name": "load_yaml",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i} FAILED: Code snippet threw an exception during execution: {e}")
                    results.append({
                        "function_name": "load_yaml",
                        "code": code,
                        "result": "failed",
                        "reason": f"Exception during execution: {e}"
                    })
                    continue  # Skip to next code snippet

        # ------------------- Write Results -------------------
        # Read existing test_result.jsonl if exists
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_records.append(json.loads(line))

        # Remove old entries for 'load_yaml'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "load_yaml"
        ]

        # Append the new results
        existing_records.extend(results)

        # Write back to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()