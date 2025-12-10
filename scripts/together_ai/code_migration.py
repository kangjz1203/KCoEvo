"""
测试llama3-70B的bock
"""
import json
import sys

from together import Together
import time
import tiktoken
from rdflib import Dataset
import os

# encoding = tiktoken.get_encoding("gpt2")



def load_json(input_file):
    """加载 JSON 文件并返回数据"""
    with open(input_file, "r") as f:
        load_dict = json.load(f)
        try:
            load_data = load_dict["data"]
        except Exception as e:
            print(e)
            print(input_file)
    return load_data

def clean_planning_res(input_str):

    input_str = input_str.strip("[]")
    # 去掉多余的字符
    json_str = input_str.strip("'```json\\n").strip("\\n```'")
    json_str = json_str.replace("\\n", "\n")
    # 解析为字典
    try:
        data_list = json.loads(json_str)['selected_paths']
    except Exception as e:
        pass
    if not input_str.strip().endswith("}"):
        json_str = json_str.rstrip() + "]}"
        try:
            data_list = json.loads(json_str)['selected_paths']
        except Exception as e:
            return json_str
    else:
        return json_str

    # 将字典放入列表中
    result = [data for data in data_list]

    return str(result)




def get_inner_kg(package, version):
    file_path = f"../../kg_data/N_Quads/{package}/{package}{version}.nq"
    res = load_kg(file_path)
    return res


def get_inter_kg(package, old_version, new_version):
    file_path = f"../kg_data/N_Quads/{package}/changes/{old_version}_{new_version}_changes.nq"
    res = load_kg(file_path)
    return res


def load_kg(file_path):
    # 创建一个 RDF 图
    graph = Dataset()

    # 加载 .nq 文件
    graph.parse(file_path, format="nquads")

    # 提取前三个三元组（忽略 graph）
    triples = []
    for subject, predicate, obj, _ in graph.quads():
        # 将 URIRef 或 Literal 转换为字符串
        subject_str = str(subject)
        predicate_str = str(predicate)
        obj_str = str(obj)
        triples.append((subject_str, predicate_str, obj_str))
    return triples


def truncate_text(text, max_tokens):
    # 获取GPT-3.5或GPT-4的分词器
    encoding = tiktoken.get_encoding("gpt2")
    disallowed_special = ()

    # 将文本编码为tokens
    tokens = encoding.encode(text, disallowed_special=disallowed_special)
    print(len(tokens))

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    # 将截断的tokens解码为文本
    truncated_text = encoding.decode(tokens)

    return truncated_text


def predict(text: str, model_name: str,n:int):
    """
    获取困惑度
    :param text:
    :return:
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": text}],
        frequency_penalty=0.1,
        max_tokens=512,
        logit_bias=None,
        logprobs=None,
        n=n,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        temperature=0.8,
        top_p=0.95
    )
    # content = response
    # content1 = response.choices
    choices_list = response.choices

    ans_list = []
    for c in choices_list:
        content = c.message.content
        # if "," in content:
        #     content = content.split(',')[0]
        ans_list.append(content)
    final_ans = str(ans_list)

    return final_ans




def bulid_prompt_plan(old_version, new_version, old_code_snippet, old_knowledge_graph, new_knowledge_graph, api_change_graph) -> str:
    prompt_plan = (
        f"""
        ### Task
        You need to select some paths from the knowledge graphs of both versions and a knowledge graph recording the evolution across two versions that will be helpful for the subsequent code migration task. \n I will provide you with old version number, new version number, old version code snippet, and three knowledge graphs. 
        This process is called planning.
        In the planning phase, you need to select some paths from the provided knowledge graphs that will be helpful for the subsequent code migration task. These paths should clearly reflect the API changes between the two versions and provide guidance for the code migration.\n
        By analyzing the knowledge graphs of both versions, identify paths related to the old version code snippet and provide guidance for the subsequent code migration task. You can overlook the prefix of entities and relations for betther understanding.

        ### Context
        You will receive:\n
        - Old version: {old_version}
        - New version: {new_version}
        - Old code snippet to be migrated: {old_code_snippet}
        - Knowledge graph of the old version: {old_knowledge_graph}
        - Knowledge graph of the new version: {new_knowledge_graph}
        - Knowledge graph of API changes between the two versions: {api_change_graph}

        ### Output
        Please return a dictionary containing the following and omit any explanations or extra information\n\n":
        1. "selected_paths": A list of selected paths from the three knowledge graphs.
        Your response:
        """
    )
    return prompt_plan

def bulid_prompt_reason(old_version, new_version, old_code_snippet, selected_paths) -> str:
    """

    :param old_version:
    :param new_version:
    :param old_code_snippet:
    :param old_knowledge_graph:
    :param new_knowledge_graph:
    :param selected_paths:
    :return:
    """
    prompt_reason = (
        f"""
    Task Description:
    In the planning phase, you have selected paths from the knowledge graphs that are related to the old version code snippet. \n
    Now, you need to use these paths to generate code that is compatible with the new version APIs. This process is called reasoning.
    Objective:
    By analyzing the paths selected in the planning phase, generate a code snippet that is compatible with the new version API while ensuring that the functionality and semantics remain consistent with the old version.

    Input:
    1. Old version number: {old_version}
    2. New version number: {new_version}
    3. Old version code snippet: {old_code_snippet}
    4. Selected paths from the planning phase: {selected_paths}

    Output:
    1. Code snippet compatible with the new version API.

    Provide your response as follows:\n"
    "   - Return only the code that fills the <block_mask> to complete the function for the target library version\n"
    "   - Enclose your code with <start> <end> to denote it as a Python code block\n"
    "   - Omit any explanations or extra information\n\n"
    Your response:
    """
    )

    return prompt_reason


def safe_predict(prompt, model_name, n, max_retries=10000, delay=10):
    for _ in range(max_retries):
        try:
            return predict(prompt, model_name,n)
        except Exception as e:
            print(f"错误：{e}。{delay} 秒后重试...")
            time.sleep(delay)
    print("达到最大重试次数。无法处理此项。")
    return None


if __name__ == "__main__":

    max_tokens = 8000  # llama3-8b窗口8k
    client = Together(api_key='159c9dc94810155413c9c4c7b2022eb232fc90e0d8f781d382604d029fd51b4b')
    # model_name = "deepseek-ai/DeepSeek-V3"
    # model_name = "Qwen/Qwen2.5-7B-Instruct-Turbo"
    model_name = sys.argv[1]
    test_file_dir = "../../datasets/code_migration/samples"
    model_name_file = model_name.split("/")[-1]
    output_dir = os.path.join(test_file_dir, "outputs", model_name_file)
    index = 'all'

    edit_orders = ['major_to_major','major_to_minor','minor_to_major','minor_to_minor']

    for edit_order in edit_orders:

        print(f"Edit_order...{edit_order}")

        original = 'old'  # migration参考的库版本
        target = 'new'  # 目标库版本



        file_name = f'{edit_order}.json'
        input_file_path = os.path.join(test_file_dir, file_name)

        with open(input_file_path, 'r', encoding='utf-8') as fr:
            lodict = json.load(fr)
        data_dict = lodict

        #   逐条预测

        for data in data_dict['data']:
            if "model_output" in data:
                print(f"第{data_dict['data'].index(data) + 1}条已经预测过，跳过该数据！")
                continue

            print(f"正在预测第{data_dict['data'].index(data) + 1}条")
            dependency = data['dependency']

            # without dependency
            old_version = data[f'{original}_version']

            new_version = data[f'{target}_version']

            original_version = data['dependency'] + data[f'{original}_version']
            target_version = data['dependency'] + data[f'{target}_version']

            original_code = data[f'{original}_code']
            masked_code = data[f'{target}_code']

            old_kg = get_inner_kg(dependency, old_version)
            new_kg = get_inner_kg(dependency, new_version)
            evol_kg = get_inter_kg(dependency, old_version, new_version)
            print("planning....")
            instruction_p = bulid_prompt_plan(original_version, target_version, original_code, old_kg, new_kg, evol_kg)
            truncated_text_p = truncate_text(instruction_p, max_tokens)
            planning_prediction = safe_predict(truncated_text_p, model_name,1)



            clean_planning_prediction = clean_planning_res(planning_prediction)



            selected_paths = clean_planning_prediction
            print("reasoning....")
            instruction_r = bulid_prompt_reason(original_version, target_version, original_code,selected_paths)
            truncated_text_r = truncate_text(instruction_r, max_tokens)
            reasoning_prediction = safe_predict(truncated_text_r, model_name,6)
            data["selected_path"] = selected_paths
            data['model_output'] = reasoning_prediction

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w', encoding='utf-8') as fw:
            json.dump(data_dict, fw, indent=4, ensure_ascii=False)

        print(f"Write to {output_file}")
        print("Done!")

        # prompt_migration = (
        #     f"""
        #     planning 阶段的提示词
        #
        #     任务描述：
        #     你需要根据提供的旧版本号、新版本号、旧版本代码片段以及两个版本的知识图谱，选择一些对后续 code migration 任务有帮助的 paths。这个过程称为 planning。
        #
        #     输入：
        #     1. 旧版本号：{old_version}
        #     2. 新版本号：{new_version}
        #     3. 旧版本代码片段：{old_code_snippet}
        #     4. 旧版本知识图谱：{old_knowledge_graph}
        #     5. 新版本知识图谱：{new_knowledge_graph}
        #
        #     输出：
        #     1. 从知识图谱中选择的 paths，这些 paths 对 code migration 任务有帮助。
        #
        #     目标：
        #     通过分析两个版本的知识图谱，找到与旧版本代码片段相关的 paths，并为后续的 code migration 任务提供指导。
        #     """
        #
        # )

        # # V1
        # prompt_migration = (
        #     "You are a Python engineer tasked with code migration. Your job is to update a given Python code snippet to make it compatible with a new version of the specified library by filling in the masked code block. Here are your instructions:\n"
        #     "1. You will receive:\n"
        #     "   - The original library version and its complete code snippet\n"
        #     "   - The target library version with a code snippet that has a <block_mask> for the parts that need updates\n\n"
        #     "2. Based on the provided information, replace the <block_mask> in the target code snippet to make it compatible with the target library version.\n\n"
        #     "3. Provide your response as follows:\n"
        #     "   - Return only the code that fills the <block_mask> to complete the function for the target library version\n"
        #     "   - Enclose your code with ```python and ``` to denote it as a Python code block\n"
        #     "   - Omit any explanations or extra information\n\n"
        #     "Here is the information and code snippet you need to work on:\n"
        #     f"Original Library and Version:\n{original_version}\n\n"
        #     f"Original Code Snippet:\n{original_code}\n\n"
        #     f"Target Library and Version:\n{target_version}\n\n"
        #     f"Target Code Snippet with <block_mask>:\n{masked_code}\n\n"
        #     "Your response:"
        # )

        # # block-level
        # prompt = (
        #     "You are a professional Python engineer. Your task is to write Python code that implements a specific function based on the provided library and version. Here are your instructions:\n"
        #     "1. You will receive:\n"
        #     "   - The name and version of the library relevant to the code\n"
        #     "   - A code snippet with a <block_mask> where you need to infer the missing code\n\n"
        #     "2. Based on the library information, write the Python code that fills the <block_mask> and implements the feature.\n\n"
        #     "3. Provide your response as follows:\n"
        #     "   - Return only the code that fills the <block_mask> and implements the function\n"
        #     "   - Enclose your code with ```python and ``` to denote it as a Python code block\n"
        #     "   - Omit any explanations or extra information\n\n"
        #     "The library information and partially masked code snippet are provided below:\n"
        #     f"Library and Version:\n{dependency_version}\n\n"
        #     f"Code Snippet with <block_mask>:\n{masked_code}\n\n"
        #     "Your response:"
        # )

        # # block-level, without version
        # prompt = (
        #     "You are a professional Python engineer. Your task is to write Python code that implements a specific function based on the provided library. Here are your instructions:\n"
        #     "1. You will receive:\n"
        #     "   - The name of the library relevant to the code\n"
        #     "   - A code snippet with a <block_mask> where you need to infer the missing code\n\n"
        #     "2. Based on the library information, write the Python code that fills the <block_mask> and implements the feature.\n\n"
        #     "3. Provide your response as follows:\n"
        #     "   - Return only the code that fills the <block_mask> and implements the function\n"
        #     "   - Enclose your code with ```python and ``` to denote it as a Python code block\n"
        #     "   - Omit any explanations or extra information\n\n"
        #     "The library information and partially masked code snippet are provided below:\n"
        #     f"Library:\n{dependency}\n\n"
        #     f"Code Snippet with <block_mask>:\n{masked_code}\n\n"
        #     "Your response:"
        # )