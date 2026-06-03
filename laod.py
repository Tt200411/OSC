# 导入NumPy库
import numpy as np

# 定义.npy文件的路径（替换成你自己的文件路径）
file_path = "informer_ETTh1_gelu/metrics.npy"  # 示例：Windows路径 "C:/data/test.npy"，Linux/Mac路径 "/home/user/data/test.npy"

try:
    # 加载.npy文件
    data = np.load(file_path)
    
    # 打印加载的数据，验证是否成功
    print("成功加载.npy文件！")
    print("文件中的数据：")
    print(data)
    # 可选：打印数据的形状和类型，方便了解数据结构
    print(f"数据形状：{data.shape}")
    print(f"数据类型：{data.dtype}")
    
except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}，请检查文件路径是否正确！")
except Exception as e:
    print(f"加载文件时出错：{e}")