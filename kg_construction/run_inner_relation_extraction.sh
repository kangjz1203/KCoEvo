#!/bin/bash

# 定义要处理的包名列表
packages=("datasets" "imageio" "jedi" "keras" "markupsafe" "pandas" "pyrsistent" "pytorch_lightning" "streamlit" "tensorflow" "torch")   # 替换为实际的包名

# 循环遍历包名列表
for package_name in "${packages[@]}"; do
    echo "Processing package: $package_name"

    # 执行 Python 脚本
    python inner_relation_extraction.py "$package_name"

    # 检查命令是否成功
    if [ $? -eq 0 ]; then
        echo "Successfully processed package: $package_name"
    else
        echo "Failed to process package: $package_name"
    fi

    echo "----------------------------------------"
done

echo "All packages processed."