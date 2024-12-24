import unittest
import json
import sys
import re
import os
from typing import Any, Dict

TEST_RESULT_JSONL = "test_result.jsonl"

_DATAPIPELINE = {}

def register_datapipeline(name):
    """Decorator used to register a CARP architecture"""
    def register_class(cls, name):
        _DATAPIPELINE[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls


class TRLConfig:
    """Mock TRLConfig class for testing purposes"""
    def __init__(self, method, model, tokenizer, optimizer, scheduler, train):
        self.method = method
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train = train

    @classmethod
    def from_dict(cls, config: Dict):
        """Converts a dictionary to a TRLConfig object"""
        return cls(
            method=config["method"],
            model=config["model"],
            tokenizer=config["tokenizer"],
            optimizer=config["optimizer"],
            scheduler=config["scheduler"],
            train=config["train"]
        )

# Mocking other classes used in the original from_dict function
class ModelConfig:
    @classmethod
    def from_dict(cls, config: Dict):
        return {"model_type": config["type"]}

class TokenizerConfig:
    @classmethod
    def from_dict(cls, config: Dict):
        return {"tokenizer_type": config["type"]}

class OptimizerConfig:
    @classmethod
    def from_dict(cls, config: Dict):
        return {"optimizer_type": config["type"]}

class SchedulerConfig:
    @classmethod
    def from_dict(cls, config: Dict):
        return {"scheduler_type": config["type"]}

class TrainConfig:
    @classmethod
    def from_dict(cls, config: Dict):
        return {"epochs": config["epochs"], "batch_size": config["batch_size"]}

# 修复 get_method 方法，返回包含 from_dict 的对象
def get_method(name: str):
    """模拟 get_method 函数，返回一个包含 from_dict 方法的对象"""
    # 这里简单地返回一个有 `from_dict` 方法的对象
    class Method:
        @staticmethod
        def from_dict(config):
            return {"method_name": config["name"]}

    return Method()

class TestFromDict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file (CarperAI_trlx_result.json) and read the 4th entry
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.config_data = data[3]  # Get the 4th JSON element (index 3)
        if len(cls.config_data) < 1:
            raise ValueError("Expected at least one configuration in the 4th JSON array")

    def test_from_dict(self):
        """Dynamically test the 'from_dict' method with sample configurations."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, config in enumerate(self.config_data):
            with self.subTest(config_index=i):
                print(f"Running test for config {i}...")
                
                # 1) 静态检查：判断 snippet 中是否真的定义了 from_dict
                if "def from_dict" not in config:
                    print(f"Config {i}: FAILED, 'from_dict' function definition not found.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "from_dict",
                        "config": config,
                        "result": "failed"
                    })
                    continue
                
                func_pattern = r"def\s+from_dict\s*\("
                if not re.search(func_pattern, config):
                    print(f"Config {i}: FAILED, incorrect signature for 'from_dict'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "from_dict",
                        "config": config,
                        "result": "failed"
                    })
                    continue

                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'sys': sys,
                    '_DATAPIPELINE': {},
                    'Any': Any,  # 注入 Any
                    'Dict': Dict,  # 注入 Dict
                    'TRLConfig': TRLConfig,  # 注入 mock TRLConfig class
                    'get_method': get_method,  # 注入模拟的 get_method 函数
                    'ModelConfig': ModelConfig,  # 注入 ModelConfig
                    'TokenizerConfig': TokenizerConfig,  # 注入 TokenizerConfig
                    'OptimizerConfig': OptimizerConfig,  # 注入 OptimizerConfig
                    'SchedulerConfig': SchedulerConfig,  # 注入 SchedulerConfig
                    'TrainConfig': TrainConfig  # 注入 TrainConfig
                }
                exec_locals = {}

                try:
                    # 动态执行配置代码片段
                    exec(config, exec_globals, exec_locals)

                    # 确保 'from_dict' 方法属于 TRLConfig 类
                    if 'from_dict' not in exec_locals or not callable(exec_locals['from_dict']):
                        print(f"Config {i}: FAILED, 'from_dict' method not found or not callable.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "from_dict",
                            "config": config,
                            "result": "failed"
                        })
                        continue

                    # 模拟字典数据
                    sample_config = {
                        "method": {"name": "sample_method"},
                        "model": {"type": "sample_model"},
                        "tokenizer": {"type": "sample_tokenizer"},
                        "optimizer": {"type": "sample_optimizer"},
                        "scheduler": {"type": "sample_scheduler"},
                        "train": {"epochs": 10, "batch_size": 32},
                    }

                    # 使用 from_dict 方法进行测试
                    result = exec_locals['from_dict'](TRLConfig, sample_config)
                    
                    # 如果从字典中创建的实例没有抛出异常，那么通过
                    self.assertIsInstance(result, TRLConfig, f"Config {i}: from_dict did not return an instance of TRLConfig.")
                    
                    # 输出测试通过
                    print(f"Config {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "from_dict",
                        "config": config,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Config {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "from_dict",
                        "config": config,
                        "result": "failed"
                    })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.config_data)}\n")
        self.assertEqual(passed_count + failed_count, len(self.config_data), "Test count mismatch!")

        # ============= 将测试结果写入 test_result.jsonl =============
        # 读取现有 test_result.jsonl（若不存在则忽略）
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # 删除 function_name == "from_dict" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "from_dict"
        ]

        # 将新结果附加
        existing_records.extend(results)

        # 重写 test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()