import json
import sys
import time
import tiktoken
from rdflib import Dataset
import os
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
# 加载本地模型
# 设置使用 GPU 6
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
# 设置内存分配器配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

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
def clean_gpu_memory():
    """清理 GPU 内存"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def clean_planning_res(input_str):
    input_str = input_str.strip("[]")
    json_str = input_str.strip("'```json\\n").strip("\\n```'")
    json_str = json_str.replace("\\n", "\n")
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
    graph = Dataset()
    graph.parse(file_path, format="nquads")
    triples = []
    for subject, predicate, obj, _ in graph.quads():
        subject_str = str(subject)
        predicate_str = str(predicate)
        obj_str = str(obj)
        triples.append((subject_str, predicate_str, obj_str))
    return triples

def truncate_text(text, max_tokens):
    encoding = tiktoken.get_encoding("gpt2")
    disallowed_special = ()
    tokens = encoding.encode(text, disallowed_special=disallowed_special)
    print(len(tokens))
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(tokens)
    return truncated_text



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

def bulid_prompt_reason(old_version, new_version, old_code_snippet, masked_code, selected_paths) -> str:
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
    5. Target Code Snippet with <block_mask>:\n{masked_code}\n\n"

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

def predict(text: str, num_return: int = 1):
    inputs = tokenizer(text, return_tensors="pt", max_length=8000, truncation=True)
    inputs = inputs.to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        num_return_sequences=num_return,  # 直接指定生成数量
        temperature=0.8,
        top_p=0.95,
        do_sample=True,  # 启用采样以生成多样化结果
        pad_token_id=tokenizer.eos_token_id  # 避免警告
    )

    prompt_length = inputs.input_ids.shape[1]
    ans_list = [
        tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
        for output in outputs
    ]
    return ans_list

def safe_predict(text: str, max_tokens: int, num_return: int):
    """直接返回列表，支持批量生成"""
    for _ in range(3):
        try:
            return str(predict(text, num_return=num_return))
        except torch.cuda.OutOfMemoryError:
            clean_gpu_memory()
            time.sleep(5)
        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(2)
    return []  # 返回空列表而非占位符


if __name__ == "__main__":
    try:
        # 设置较小的最大 token 数以减少内存使用
        max_tokens = 4000  # 减小从 8000 到 4000
        model_name = sys.argv[1]
        # local_model_path = f"/data2/kjz/CodeKG/fine-tuned/{model_name}/saved_models"
        local_model_path = f"/datanfs2/chenyongrui/huggingface/{model_name}"

        # 配置量化参数
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # 加载模型和分词器
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            device_map="auto",
            # quantization_config=bnb_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )



        test_file_dir = "../../datasets/code_migration/samples"
        model_name_file = model_name.split("/")[-1]
        output_dir = os.path.join(test_file_dir, "outputs", model_name_file)
        index = 'all'
        edit_orders = ['old_to_new', 'new_to_old']

        for edit_order in edit_orders:
            print(f"edit order....{edit_order}")
            original = 'old'  # migration参考的库版本
            target = 'new'  # 目标库版本

            if edit_order == 'new_to_old':
                original = 'new'
                target = 'old'

            file_name = f'code_migration_exe_{edit_order}.json'
            input_file_path = os.path.join(test_file_dir, file_name)

            with open(input_file_path, 'r', encoding='utf-8') as fr:
                lodict = json.load(fr)
            data_dict = lodict

            for data in data_dict['data']:
                if "model_output" in data:
                    print(f"第{data_dict['data'].index(data) + 1}条已经预测过，跳过该数据！")
                    continue

                print(f"正在预测第{data_dict['data'].index(data) + 1}条")
                dependency = data['dependency']
                old_version = data[f'{original}_version']
                new_version = data[f'{target}_version']
                original_version = data['dependency'] + data[f'{original}_version']
                target_version = data['dependency'] + data[f'{target}_version']
                original_code = data[f'{original}_task_function']
                masked_code = data[f'masked_{target}_task_function']

                old_kg = get_inner_kg(dependency, old_version)
                new_kg = get_inner_kg(dependency, new_version)
                evol_kg = get_inter_kg(dependency, old_version, new_version)
                print("planning....")
                instruction_p = bulid_prompt_plan(original_version, target_version, original_code, old_kg, new_kg, evol_kg)
                truncated_text_p = truncate_text(instruction_p, max_tokens)
                planning_prediction = str(safe_predict(truncated_text_p, model_name, 1))

                clean_planning_prediction = clean_planning_res(planning_prediction)
                selected_paths = clean_planning_prediction


                print("reasoning....")
                instruction_r = bulid_prompt_reason(original_version, target_version, original_code, masked_code, selected_paths)
                truncated_text_r = truncate_text(instruction_r, max_tokens)
                reasoning_prediction = safe_predict(truncated_text_r, model_name, 6)
                data['model_output'] = str(reasoning_prediction)
                data["selected_path"] = str(selected_paths)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, file_name)
            with open(output_file, 'w', encoding='utf-8') as fw:
                json.dump(data_dict, fw, indent=4, ensure_ascii=False)

            print(f"Write to {output_file}")
            print("Done!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    finally:
        # 确保在程序结束时清理内存
        clean_gpu_memory()