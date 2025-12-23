#!/bin/bash

# 定义允许的模型列表
allowed_models=(
    "Qwen/Qwen2.5-7B-Instruct-Turbo"
    "deepseek-ai/DeepSeek-V3"
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo"
    "Qwen/Qwen2.5-Coder-32B-Instruct"
    "claude3.5-sonnet"
    "gpt-3.5-turbo"
    "Qwen2.5-Coder-7B-Instruct"
    "ft-Qwen2.5-Coder-7B-Instruct"
    "mistralai/Mistral-Small-24B-Instruct-2501"
    "gpt-4o"
    "Meta-Llama-3-8B-Instruct"
    "ft-Meta-Llama-3-8B-Instruct"
    "deepseek-coder-7b-base-v1.5"
    "deepseek-coder-7b-base-v1.5"
    "ft-deepseek-coder-7b-base-v1.5"
)

# 检查是否提供了足够的参数
if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage: $0 model_name ft_type"
    echo "This script will run evaluation for all edit_order combinations and calculate averages."
    echo "Allowed models are: ${allowed_models[*]}"
    exit 1
fi

# 获取参数
model_name=$1
ft_type=$2

# 定义edit_order组合
declare -a edit_orders_group1=("major major" "major minor" "minor major" "minor minor")
declare -a edit_orders_group2=("new old" "old new")

# 初始化累计值数组
declare -a group1_cdc1_values=()
declare -a group1_cdc3_values=()
declare -a group1_em1_values=()

declare -a group2_cdc1_values=()
declare -a group2_cdc3_values=()
declare -a group2_em1_values=()

# 函数：运行单次评估并提取指标
run_single_evaluation() {
    local model_name=$1
    local ft_type=$2
    local edit_order_1=$3
    local edit_order_2=$4

    echo "Running evaluation for: $model_name $ft_type $edit_order_1 $edit_order_2"

    # 运行clear_ans_update.py
    echo "Running clear_ans_update.py..."
    if ! python clear_ans_update.py "$model_name" "$ft_type" "$edit_order_1" "$edit_order_2"; then
        echo "Error: Failed to execute clear_ans_update.py with exit code $?"
        echo "Command was: python clear_ans_update.py \"$model_name\" \"$ft_type\" \"$edit_order_1\" \"$edit_order_2\""
        return 1
    fi

    # 运行choose_core_line_from_block_versicode.py
    echo "Running choose_core_line_from_block_versicode.py..."
    if ! python choose_core_line_from_block_versicode.py "$model_name" "$ft_type" "$edit_order_1" "$edit_order_2"; then
        echo "Error: Failed to execute choose_core_line_from_block_versicode.py with exit code $?"
        echo "Command was: python choose_core_line_from_block_versicode.py \"$model_name\" \"$ft_type\" \"$edit_order_1\" \"$edit_order_2\""
        return 1
    fi

    # 运行compute_migration_cdc.py并捕获输出
    echo "Running compute_migration_cdc.py..."
    local output=$(python compute_migration_cdc.py "$model_name" "$ft_type" "$edit_order_1" "$edit_order_2" 2>&1)
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Error: Failed to execute compute_migration_cdc.py with exit code $exit_code"
        echo "Command was: python compute_migration_cdc.py \"$model_name\" \"$ft_type\" \"$edit_order_1\" \"$edit_order_2\""
        echo "Output was:"
        echo "$output"
        return 1
    fi

    # 从输出中提取指标 - 使用新的简化格式
    echo "Output from compute_migration_cdc.py:"
    echo "$output"

    # 提取简化格式的输出行：SHELL_PARSE:cdc1,cdc3,em1
    local metrics_line=$(echo "$output" | grep "SHELL_PARSE:" | head -1)

    if [ -z "$metrics_line" ]; then
        echo "Error: Could not find SHELL_PARSE line in output"
        echo "Full output:"
        echo "$output"
        return 1
    fi

    # 提取数值部分（去掉 SHELL_PARSE: 前缀）
    local metrics_values=$(echo "$metrics_line" | sed 's/SHELL_PARSE://')

    # 使用逗号分割提取各个指标
    local cdc1=$(echo "$metrics_values" | cut -d',' -f1)
    local cdc3=$(echo "$metrics_values" | cut -d',' -f2)
    local em1=$(echo "$metrics_values" | cut -d',' -f3)

    echo "Extracted metrics: CDC@1='$cdc1', CDC@3='$cdc3', EM@1='$em1'"

    # 验证提取的值是否有效
    if [[ ! "$cdc1" =~ ^[0-9]*\.?[0-9]+$ ]] || [[ ! "$cdc3" =~ ^[0-9]*\.?[0-9]+$ ]] || [[ ! "$em1" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        echo "Error: Failed to extract valid numeric metrics"
        echo "CDC@1: '$cdc1', CDC@3: '$cdc3', EM@1: '$em1'"
        echo "Metrics line was: '$metrics_line'"
        echo "Full output:"
        echo "$output"
        return 1
    fi

    # 只返回数值，不返回调试信息
    echo "$cdc1 $cdc3 $em1"
}

echo "Starting evaluation pipeline for model: $model_name, ft_type: $ft_type"
echo "============================================================================"

# 处理第一组 (major-major, major-minor, minor-major, minor-minor)
echo "Processing Group 1: major/minor combinations..."
for edit_pair in "${edit_orders_group1[@]}"; do
    read -r edit_order_1 edit_order_2 <<< "$edit_pair"
    echo "Processing edit pair: '$edit_order_1' to '$edit_order_2'"

    result=$(run_single_evaluation "$model_name" "$ft_type" "$edit_order_1" "$edit_order_2")
    exit_status=$?

    if [ $exit_status -eq 0 ]; then
        echo "Debug: Raw result from run_single_evaluation: '$result'" >&2

        # 提取result的最后一行（只包含数值的行）
        result_clean=$(echo "$result" | tail -1)
        echo "Debug: Cleaned result: '$result_clean'" >&2

        read -r cdc1 cdc3 em1 <<< "$result_clean"
        echo "Debug: Parsed values - CDC@1: '$cdc1', CDC@3: '$cdc3', EM@1: '$em1'" >&2

        # 验证解析出的值是否为有效数字
        if [[ "$cdc1" =~ ^[0-9]*\.?[0-9]+$ ]] && [[ "$cdc3" =~ ^[0-9]*\.?[0-9]+$ ]] && [[ "$em1" =~ ^[0-9]*\.?[0-9]+$ ]]; then
            group1_cdc1_values+=("$cdc1")
            group1_cdc3_values+=("$cdc3")
            group1_em1_values+=("$em1")
            echo "Debug: Added valid values to Group 1 arrays" >&2
        else
            echo "Error: Invalid parsed values - CDC@1: '$cdc1', CDC@3: '$cdc3', EM@1: '$em1'" >&2
            exit 1
        fi
    else
        echo "Failed to process '$edit_order_1' to '$edit_order_2' with exit status $exit_status"
        echo "Debug: Result was: '$result'"
        exit 1
    fi
    echo "----------------------------------------"
done

# 处理第二组 (new-old, old-new)
echo "Processing Group 2: new/old combinations..."
for edit_pair in "${edit_orders_group2[@]}"; do
    read -r edit_order_1 edit_order_2 <<< "$edit_pair"
    echo "Processing edit pair: '$edit_order_1' to '$edit_order_2'"

    result=$(run_single_evaluation "$model_name" "$ft_type" "$edit_order_1" "$edit_order_2")
    exit_status=$?

    if [ $exit_status -eq 0 ]; then
        echo "Debug: Raw result from run_single_evaluation: '$result'" >&2

        # 提取result的最后一行（只包含数值的行）
        result_clean=$(echo "$result" | tail -1)
        echo "Debug: Cleaned result: '$result_clean'" >&2

        read -r cdc1 cdc3 em1 <<< "$result_clean"
        echo "Debug: Parsed values - CDC@1: '$cdc1', CDC@3: '$cdc3', EM@1: '$em1'" >&2

        # 验证解析出的值是否为有效数字
        if [[ "$cdc1" =~ ^[0-9]*\.?[0-9]+$ ]] && [[ "$cdc3" =~ ^[0-9]*\.?[0-9]+$ ]] && [[ "$em1" =~ ^[0-9]*\.?[0-9]+$ ]]; then
            group2_cdc1_values+=("$cdc1")
            group2_cdc3_values+=("$cdc3")
            group2_em1_values+=("$em1")
            echo "Debug: Added valid values to Group 2 arrays" >&2
        else
            echo "Error: Invalid parsed values - CDC@1: '$cdc1', CDC@3: '$cdc3', EM@1: '$em1'" >&2
            exit 1
        fi
    else
        echo "Failed to process '$edit_order_1' to '$edit_order_2' with exit status $exit_status"
        echo "Debug: Result was: '$result'"
        exit 1
    fi
    echo "----------------------------------------"
done

# 改进的计算平均值函数 - 兼容老版本 bash
calculate_average() {
    local values_string="$1"
    local sum=0
    local count=0
    local value

    echo "Debug: Calculating average for values: $values_string" >&2

    # 将字符串按空格分割处理
    for value in $values_string; do
        # 检查值是否为有效数字
        if [[ "$value" =~ ^[0-9]*\.?[0-9]+$ ]] && [ -n "$value" ]; then
            sum=$(awk "BEGIN {print $sum + $value}")
            ((count++))
            echo "Debug: Added $value, sum=$sum, count=$count" >&2
        else
            echo "Debug: Skipping invalid value: '$value'" >&2
        fi
    done

    if [ $count -gt 0 ]; then
        awk "BEGIN {printf \"%.6f\", $sum / $count}"
    else
        echo "0.000000"
    fi
}

# 计算第一组的平均值
echo "Calculating Group 1 averages..."
group1_avg_cdc1=$(calculate_average "${group1_cdc1_values[*]}")
group1_avg_cdc3=$(calculate_average "${group1_cdc3_values[*]}")
group1_avg_em1=$(calculate_average "${group1_em1_values[*]}")

# 计算第二组的平均值
echo "Calculating Group 2 averages..."
group2_avg_cdc1=$(calculate_average "${group2_cdc1_values[*]}")
group2_avg_cdc3=$(calculate_average "${group2_cdc3_values[*]}")
group2_avg_em1=$(calculate_average "${group2_em1_values[*]}")

# 输出最终结果
echo "============================================================================"
echo "FINAL RESULTS FOR MODEL: $model_name, FT_TYPE: $ft_type"
echo "============================================================================"
echo "Group 1 (major/minor) Average Results:"
echo "  Average CDC@1: $group1_avg_cdc1"
echo "  Average CDC@3: $group1_avg_cdc3"
echo "  Average EM@1:  $group1_avg_em1"
echo ""
echo "Group 2 (new/old) Average Results:"
echo "  Average CDC@1: $group2_avg_cdc1"
echo "  Average CDC@3: $group2_avg_cdc3"
echo "  Average EM@1:  $group2_avg_em1"
echo "============================================================================"

echo "Pipeline execution completed successfully."