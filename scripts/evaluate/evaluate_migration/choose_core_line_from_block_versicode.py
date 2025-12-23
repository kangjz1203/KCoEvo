"""
versicode中的block找出模型生成的那一行代码
"""
import os
import re
import json
import random
import sys


def process_line_mask(code_snippet, core_token):
    if not core_token:
        # 如果 calls 为空列表，则返回三个 None 值
        return None, None

    # 定义存储指定 API 所在行位置的字典，键是行数，值是行内容
    replaced_lines = {}
    lines = code_snippet.split("\n")

    # 标记是否处于多行注释中
    in_multi_line_comment = False

    # 检查每一行是否包含指定的 API 调用，并且不在注释中
    for i, line in enumerate(lines):
        if in_multi_line_comment:
            # 如果处于多行注释中，则检查当前行是否包含注释结束标记
            if ('"""' in line or "'''" in line) and not re.findall(r"'''(.*?)'''|\"\"\"(.*?)\"\"\"", line):
                in_multi_line_comment = False
            continue
        elif line.strip().startswith("#"):
            # 如果是单行注释，则跳过当前行
            continue
        elif re.findall(r"'''(.*?)'''|\"\"\"(.*?)\"\"\"", line):
            # 如果存在"""xxx"""或'''xxx'''的单行注释，则跳过当前行
            continue
        elif ('"""' in line or "'''" in line) and not re.findall(r"'''(.*?)'''|\"\"\"(.*?)\"\"\"", line):
            # 如果当前行包含三对双引号或三对单引号，可能为多行注释
            in_multi_line_comment = True
            continue
        else:
            # 检查是否为特定的函数定义行
            if re.search(r'\bdef\s+task_function\b', line):
                continue

            # 检查当前行是否包含指定的 API 调用，并且是否包含给定的 aliases 中的任何一项
            if re.search(r'\b{}\b(?!\s*=)'.format(re.escape(core_token)), line):
                # 将匹配到的行存储到替换列表中
                replaced_lines.update({i: line})

    # if replaced_lines:
    #     random_line_location = random.choice(list(replaced_lines.keys()))
    #
    #     # 替换当前行为 <mask>，但保留句首空格或缩进
    #     masked_line = lines[random_line_location]
    #     # 移除句首的空格或缩进
    #     leading_spaces = re.match(r'^\s*', masked_line).group(0)
    #     masked_line = masked_line.strip()
    #     lines[random_line_location] = leading_spaces + "<line_mask>"
    if replaced_lines:
        # random_line_location = random.choice(list(replaced_lines.keys())) # 旧代码
        # 总是选择第一个匹配的行（假设 replaced_lines.keys() 返回的顺序是基于行号的）
        # 为了确保是第一个，可以先对行号排序
        sorted_line_locations = sorted(list(replaced_lines.keys()))
        if sorted_line_locations:
            random_line_location = sorted_line_locations[-1]  # 选择第一个
        else:  # 理论上不应该发生，因为 replaced_lines 为真
            return None, None
        masked_line = lines[random_line_location]
        leading_spaces = re.match(r'^\s*', masked_line).group(0)
        masked_line = lines[random_line_location]
        lines[random_line_location] = leading_spaces + "<line_mask>"

        # 拼接为完整代码
        masked_code = '\n'.join(lines)

        return masked_code, masked_line

    # 如果没有找到匹配行，返回两个 None 值
    return None, None


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
# model_list = os.listdir('../../dataset/final_dataset/generate_block_result_data2024_5_14v2')

    # input_json_file = f'../../dataset/final_dataset/execute_dataset/{model}/test_data_block_with_complete_import.json'
    # output_json_file = f'../../dataset/final_dataset/execute_dataset/{model}/test_data_block_with_complete_import_temp.json'
    # input_json_file = f'../../dataset/final_dataset/generate_block_result_data2024_5_14v2/{model}/random_block_data_2024_5_14v2.json'
    # output_json_file = f'../../dataset/final_dataset/generate_block_result_data2024_5_14v2/{model}/random_block_data_2024_5_14v2_temp.json'


    model_name = sys.argv[1]
    ft_type = sys.argv[2]
    edit_order_1 = sys.argv[3]  # 原始 JSONL 文件路径
    edit_order_2 = sys.argv[4]  # 原始 JSONL 文件路径

    if "/" in model_name:
        model_file_name = model_name.split("/")[0]
    else:
        model_file_name = model_name

    output_dir = f"../../datasets/code_migration/samples/outputs/{ft_type}/{model_file_name}/res/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    input_json_file = f"../../datasets/code_migration/samples/outputs/{ft_type}/{model_file_name}/res/{edit_order_1}_to_{edit_order_2}.json"

    output_json_file = input_json_file

    data = load_json(input_json_file)

    for item in data['data']:
        # core_token = item['answer']
        # code = item['code']

        # core_token = item['core_token']
        # code = item['core_block']

        #old to new
        core_token = item['new_name']
        code = item['new_code']

        # # new to old
        # core_token = item['old_name']
        # code = item['old_code']


        # 利用core_token, 从被遮蔽的block中定位core_line
        _, core_line_in_core_block = process_line_mask(code, core_token)
        if core_line_in_core_block:
            item['core_line_in_core_block'] = core_line_in_core_block
        else:
            item['core_line_in_core_block'] = "N/A"

        model_output_clear = item['model_output_clear']
        core_line_in_output_list = []

        # for entry in eval(model_output_clear):
        for entry in model_output_clear:
            # 利用core_token, 从模型的输出中定位core_line （列表中的每个元素都要）
            _, core_line_in_output = process_line_mask(entry, core_token)
            if core_line_in_output:
                core_line_in_output_list.append(core_line_in_output)
            else:
                core_line_in_output_list.append("N/A")

        item['core_line_in_output_clear'] = core_line_in_output_list

    save_json(output_json_file, data)
    print("Done!")

