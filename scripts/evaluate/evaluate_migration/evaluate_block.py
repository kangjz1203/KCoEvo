"""
评测block的预测能力
1、判断是否包含正确的函数名
2、判断是否合法
3、计算ISM，和PM
"""
import json
import tokenize
import io
import math
import ast
import re
import sys
import os
def is_code_valid(code):

    try:
        compile(code, '<string>', 'exec')
        return True
    except:
        return False


def longest_common_prefix_between_lists_with_elements(list1, list2):
    """
    计算两个字符串列表中元素的最长前缀匹配长度
    :param list1:
    :param list2:
    :return:
    """
    max_prefix_length = 0
    max_prefix_elements = ()
    for str1 in list1:
        for str2 in list2:
            prefix_length = 0
            min_len = min(len(str1), len(str2))
            for i in range(min_len):
                if str1[i] == str2[i]:
                    prefix_length += 1
                else:
                    break
            if prefix_length > max_prefix_length:
                max_prefix_length = prefix_length
                max_prefix_elements = (str1, str2)
    return max_prefix_length, max_prefix_elements

def get_token(ans_code:str, output_code:str):
    """
    对代码进行词法分析，分解成标识符，返回两个标识符列表
    :param ans_code:
    :param output_code:
    :return:
    """
    output_flag = True
    ans_flag = True
    try:
        tokens_ans = tokenize.tokenize(io.BytesIO(ans_code.encode('utf-8')).readline)
    except Exception as e:
        tokens_ans = ans_code.splitlines()
        ans_flag = False

    try:
        tokens_output = tokenize.tokenize(io.BytesIO(output_code.encode('utf-8')).readline)
    except Exception as e:
        tokens_output = output_code.splitlines()
        output_flag = False


    identifiers_ans = []
    identifiers_output = []
    if ans_flag == True:
        try:
            for token in tokens_ans:
                if token.type == tokenize.NAME:
                    identifiers_ans.append(token.string)
        except Exception as e:
            identifiers_ans = tokens_ans
    else:
        identifiers_ans = tokens_ans

    if output_flag == True:
        try:
            for to in tokens_output:
                if to.type == tokenize.NAME:
                    identifiers_output.append(to.string)
        except Exception as e:
            identifiers_output = tokens_output
    else:
        identifiers_output = tokens_output


    return identifiers_ans, identifiers_output


def get_token_per_line(code: str):
    """
    对每一行代码进行词法分析，记录每一行的标识符
    :param code: 代码字符串
    :return: 每一行的标识符列表组成的列表
    """
    lines = code.split('\n')  # 将代码按行分割成列表
    identifiers_per_line = []  # 用于存储每一行的标识符列表的列表

    for line in lines:
        tokens = tokenize.tokenize(io.BytesIO(line.encode('utf-8')).readline)
        identifiers = []
        try:
            for token in tokens:
                if token.type == tokenize.NAME:
                    identifiers.append(token.string)
        except:
            identifiers = line.split(' ')
        identifiers_per_line.append(identifiers)

    return identifiers_per_line



def get_ISM(answer_code:str, model_output_list:list, asnwer_name:str, model_filled_code)->list:
    """
    计算ISM，返回一个有序的得分列表
    :return:
    """
    score_list = []
    for code in model_output_list:

        for index, code in enumerate(model_output_list):
            # if asnwer_name not in code or not is_code_valid(model_filled_code[str(index+1)]):
            if not re.search(rf'\b{re.escape(asnwer_name)}\b', model_filled_code[str(index+1)]) or not is_code_valid(model_filled_code[str(index + 1)]):
                score_list.append(0)
                continue

        # if asnwer_name not in code or is_code_valid(code) == False:
        #     score_list.append(0)
        #     continue

        # if asnwer_name not in code:
        #     score_list.append(0)
        #     continue

        identifiers_ans, identifiers_output = get_token(answer_code, code)
        max_len, elements = longest_common_prefix_between_lists_with_elements(identifiers_ans, identifiers_output)
        if max_len != 0:
            base_element_len = max(len(elements[0]), len(elements[1]))
            temp_score = max_len/base_element_len
            score_list.append(temp_score)
        else:
            score_list.append(0)
        # base_element_len = max(len(elements[0]), len(elements[1]))
        # temp_score = max_len/base_element_len
        # score_list.append(temp_score)

    score_list = sorted(score_list, reverse=True)
    return score_list

def get_ISM_without_verification(answer_code:str, model_output_list:list, asnwer_name:str)->list:
    """
    计算ISM，返回一个有序的得分列表
    :return:
    """
    score_list = []
    for code in model_output_list:

        if asnwer_name not in code:
            score_list.append(0)
            continue

        # if asnwer_name not in code:
        #     score_list.append(0)
        #     continue

        identifiers_ans, identifiers_output = get_token(answer_code, code)
        max_len, elements = longest_common_prefix_between_lists_with_elements(identifiers_ans, identifiers_output)
        if max_len != 0:
            base_element_len = max(len(elements[0]), len(elements[1]))
            temp_score = max_len/base_element_len
            score_list.append(temp_score)
        else:
            score_list.append(0)
        # base_element_len = max(len(elements[0]), len(elements[1]))
        # temp_score = max_len/base_element_len
        # score_list.append(temp_score)

    score_list = sorted(score_list, reverse=True)
    return score_list

def longest_common_prefix_with_lengths(list1, list2):
    """
    计算两个二维列表中每个子列表的最长前缀匹配长度，并记录拥有最长前缀匹配长度的两个子列表的长度
    :param list1: 第一个二维列表
    :param list2: 第二个二维列表
    :return: 最长前缀匹配长度以及拥有最长前缀匹配长度的两个子列表的长度
    """
    max_length = 0
    len_list1 = 0
    len_list2 = 0
    for i, sublist1 in enumerate(list1):
        for j, sublist2 in enumerate(list2):
            match_length = 0
            min_length = min(len(sublist1), len(sublist2))
            for k in range(min_length):
                if sublist1[k] == sublist2[k]:
                    match_length += 1
                else:
                    break
            if match_length > max_length:
                max_length = match_length
                len_list1 = len(sublist1)
                len_list2 = len(sublist2)
    return max_length, len_list1, len_list2


def get_PM(answer_code:str, model_output_list:list, asnwer_name:str, model_filled_code)->list:
    """
    计算PM，返回一个有序的得分列表
    :return:
    """
    score_list = []
    for code in model_output_list:

        for index, code in enumerate(model_output_list):
            # if asnwer_name not in code or not is_code_valid(model_filled_code[str(index+1)]):
            if not re.search(rf'\b{re.escape(asnwer_name)}\b', model_filled_code[str(index+1)]) or not is_code_valid(model_filled_code[str(index + 1)]):
                score_list.append(0)
                continue

        # if asnwer_name not in code or is_code_valid(code) == False:
        #     score_list.append(0)
        #     continue

        # if asnwer_name not in code:
        #     score_list.append(0)
        #     continue

        ans_list = get_token_per_line(answer_code)
        output_token_list = get_token_per_line(code)
        max_len, len1, len2 = longest_common_prefix_with_lengths(ans_list, output_token_list)
        base_element_len = max(len1, len2)

        if base_element_len != 0:
            temp_score = max_len/base_element_len
            score_list.append(temp_score)
        else:
            score_list.append(0)

    score_list = sorted(score_list, reverse=True)
    return score_list

def get_score(score_list:list, k):
    """
    计算score@n,k
    :param score_list:
    :param k:
    :return:
    """
    n = len(score_list)
    sum = 0
    final = n-k+1
    for i in range(1, final+1):
        sum += math.comb(n-i, k-1) * score_list[i-1]

    final_score = sum/math.comb(n, k)

    return final_score


if __name__ == '__main__':
    model_name = sys.argv[1]
    ft_type = sys.argv[2]
    edit_order_1 = sys.argv[3]  # 原始 JSONL 文件路径
    edit_order_2 = sys.argv[4]  # 原始 JSONL 文件路径

    if "/" in model_name:
        model_file_name = model_name.split("/")[-1]
    else:
        model_file_name = model_name

    output_dir = f"../../datasets/code_migration/samples/outputs/{ft_type}/{model_file_name}/res"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    input_json_file = f"../../datasets/code_migration/samples/outputs/{ft_type}/{model_file_name}/res/{edit_order_1}_to_{edit_order_2}.json"

    with open(input_json_file, 'r', encoding='utf-8')as fr:
        lodict = json.load(fr)
    data_dict = lodict
    data_list = data_dict['data']
    data_len = len(data_list)
    sum_ISM = 0
    sum_PM = 0
    mode = edit_order_1


    for data in data_list:
        # k设为1的话，就只要第一个[:1]
        model_output_list = data['model_output_clear']
        model_filled_code = data['model_filled_code']
        answer_code = data[f'{mode}_core_block']

        # 去除缩进
        lines = answer_code.split('\n')  # 将代码按行分割
        answer_code_stripped = '\n'.join([line[4:] if line.startswith('    ') else line for line in lines])  # 去掉前四个空格并拼接

        answer_name = data[f'{mode}_core_token']

        ISM_score_list = get_ISM(answer_code_stripped, model_output_list, answer_name, model_filled_code)
        # ISM_score_without_verification_list = get_ISM_without_verification(answer_code, model_output_list, answer_name)     #新增
        PM_score_list = get_PM(answer_code_stripped, model_output_list, answer_name, model_filled_code)

        # ISM@1,PM@1,k改为1
        ISM_score = get_score(ISM_score_list, 1)
        PM_score = get_score(PM_score_list, 1)

        sum_ISM += ISM_score
        sum_PM += PM_score
        # print(f"ISM分数：{ISM_score}")
        # print(f"PM分数：{PM_score}")

    print(f"ISM分数：{sum_ISM/data_len}")
    print(f"PM分数：{sum_PM/data_len}")