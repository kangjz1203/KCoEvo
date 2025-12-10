#!/bin/bash

allowed_models=("Qwen2.5-Coder-7B-Instruct"
"Meta-Llama-3-8B-Instruct"
"deepseek-coder-7b-base-v1.5"
"CodeLlama-7b-Python-hf"

)

# 检查是否提供了足够的参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 model_name"
    echo "Allowed models are: ${allowed_models[*]}"
    exit 1
fi

# 获取参数
model_name=$1

# 检查 model_name 是否在允许的列表中
# shellcheck disable=SC2076
if [[ ! " ${allowed_models[*]} " =~ " ${model_name} " ]]; then
    echo "Error: The model_name '$model_name' is not allowed."
    echo "Allowed models are: ${allowed_models[*]}"
    exit 1
fi

# 获取参数
model_name=$1


# 依次执行三个 Python 脚本
echo "Running code_migration.py"
python code_migration.py "$model_name"

echo "Running code_migration_exe.py"
python code_migration_exe.py "$model_name"

echo "Pipeline execution completed."