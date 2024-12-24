import os
import subprocess

def execute_tests_in_directory(directory, start_test, end_test):
    """
    执行指定目录中从 start_test 到 end_test 的测试文件。
    """
    for i in range(start_test, end_test + 1):
        test_file = f"test{i}.py"
        test_path = os.path.join(directory, test_file)

        if os.path.exists(test_path):
            print(f"Executing {test_file} in {directory}...")
            try:
                # 调用 Python 运行测试文件
                result = subprocess.run(["python", test_path], capture_output=True, text=True)
                print(result.stdout)  # 打印测试输出
                if result.stderr:
                    print(f"Errors in {test_file}:")
                    print(result.stderr)
            except Exception as e:
                print(f"Error while executing {test_file}: {e}")
        else:
            print(f"{test_file} does not exist in {directory}, skipping.")

def execute_all_tests():
    """
    执行 CarperAI_trlx 和 lucidrains--imagen-pytorch 文件夹中的测试文件。
    """
    # 执行 CarperAI_trlx 中的测试
    carper_dir = "CarperAI_trlx"
    execute_tests_in_directory(carper_dir, 1, 46)

    # 执行 lucidrains--imagen-pytorch 中的测试
    imagen_dir = "lucidrains--imagen-pytorch"
    execute_tests_in_directory(imagen_dir, 47, 112)

    imagen_dir = "deepmind--tracr"
    execute_tests_in_directory(imagen_dir, 114, 258)

    imagen_dir = "leopard-ai--betty"
    execute_tests_in_directory(imagen_dir, 260, 294)

    imagen_dir = "google--lightweight_mmm"
    execute_tests_in_directory(imagen_dir, 296, 358)

    imagen_dir = "amazon-science--patchcore-inspection"
    execute_tests_in_directory(imagen_dir, 360, 391)

    imagen_dir = "facebookresearch--omnivore"
    execute_tests_in_directory(imagen_dir, 392, 413)
    
    imagen_dir = "maxhumber--redframes"
    execute_tests_in_directory(imagen_dir, 414, 455)

if __name__ == "__main__":
    execute_all_tests()