#!/bin/bash

# 定义 JSON 文件的路径
JSON_FILE="../kg_data/parsed_data/original_data/DLTrans/dl_train_torch.json"

# 循环直到 _dl_train_torch.json 中的 "data" 列表为空
while true; do
    # 检查 "data" 列表是否为空
    if jq -e '.data | length > 0' "$JSON_FILE" > /dev/null; then
        # 运行 load_dlTrans_torch.py
        if ! python load_dlTrans_torch.py; then
            echo "Error: Failed to run load_dlTrans_torch.py. Exiting."
            exit 1
        fi

        # 删除 "data" 列表中的第一个 item
        if ! jq 'del(.data[0])' "$JSON_FILE" > temp.json; then
            echo "Error: Failed to modify JSON file. Exiting."
            exit 1
        fi
        mv temp.json "$JSON_FILE"
    else
        # 如果 "data" 列表为空，退出循环
        echo "No more items in 'data' list. Exiting."
        break
    fi
done