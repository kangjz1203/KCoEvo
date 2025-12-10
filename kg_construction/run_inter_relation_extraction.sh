#!/bin/bash

# 定义 packages 列表
packages=("datasets" "imageio" "keras" "pandas" "streamlit" "tensorflow" "torch")

# 遍历每个 package
for package in "${packages[@]}"; do
    # 定义 changes 目录路径
    changes_dir="../kg_data/parsed_data/json_data/${package}/changes"

    # 检查目录是否存在
    if [ ! -d "$changes_dir" ]; then
        echo "Directory $changes_dir does not exist. Skipping..."
        continue
    fi

    # 遍历 changes 目录下的所有 JSON 文件
    for json_file in "$changes_dir"/*.json; do
        # 检查是否是文件
        if [ ! -f "$json_file" ]; then
            continue
        fi

        # 提取文件名（不包括路径）
        filename=$(basename "$json_file")

        # 使用正则表达式提取旧版本号和新版本号（去掉 ==）
        if [[ "$filename" =~ ==([0-9]+\.[0-9]+(\.[0-9]+)?)_==([0-9]+\.[0-9]+(\.[0-9]+)?)_changes\.json ]]; then
            old_version="${BASH_REMATCH[1]}"
            new_version="${BASH_REMATCH[3]}"

            # 调用 Python 脚本
            echo "Processing: $json_file"
            echo "Package: $package, Old Version: $old_version, New Version: $new_version"
            python inter_relation_extraction.py "$package" "$old_version" "$new_version"
        else
            echo "Skipping invalid filename format: $filename"
        fi
    done
done