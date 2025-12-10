import json
import sys
import time
import tiktoken
from rdflib import Dataset
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import gc

# 设置使用 GPU 6
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 设置内存分配器配置
# 在文件开头添加更优化的内存分配策略
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"

def clean_gpu_memory():
    """强化内存清理"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def clean_planning_res(input_str):
    json_str = input_str.replace("\\n", "\n")
    json_str = json_str.replace("```json","").replace("\\n```'","")
    json_str = json_str.replace("'","").replace("`","")

    # import ipdb
    # ipdb.set_trace()
    try:
        data_list = json.loads(json_str)[0]['selected_paths']
        json_str = str(data_list)

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

def predict(text: str, num_return: int = 1):
    inputs = tokenizer(text, return_tensors="pt", max_length=8000, truncation=True)
    inputs = inputs.to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
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




def bulid_prompt(original_version, original_code, target_version, description) -> str:
    """
    构建prompt
    :param version:
    :param description:
    :param masked_code:
    :param options:
    :return:
    """

    prompt_migration = (
        f"""
        You are now a professional Python programming engineer. I will provide you with a code snippet and a description of its functionality, 
        including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version. 
        Your task is to refactor the code using the methods provided by the specified new version and return the refactored code. 
        Please note that you only need to return the refactored code and enclose it with <start> and <end>:
        ###Functionality description of the code
        {description}
        ###Dependency and old version
        {original_version}
        ###Old version code
        {original_code}
        ###Dependency and new version
        {target_version}
        ###Refactored new code
            """
        "Your response:"
    )

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

    return prompt_migration


if __name__ == "__main__":
    try:
        # 设置较小的最大 token 数以减少内存使用
        max_tokens = 8000  # 减小从 8000 到 4000
        model_name = sys.argv[1]
        # local_model_path = f"/data2/kjz/CodeKG/fine-tuned/{model_name}/saved_models"
        local_model_path = f"/datanfs2/chenyongrui/huggingface/{model_name}"

        # 配置量化参数
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        # )

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



        test_file_dir = "../../../datasets/code_migration/samples"
        model_name_file = model_name.split("/")[-1]
        output_dir = os.path.join(test_file_dir, "vanilla", "outputs", model_name_file)

        edit_orders = ['major_to_major', 'major_to_minor', 'minor_to_major', 'minor_to_minor','old_to_new','new_to_old']

        for edit_order in edit_orders:
            print(f"Edit_order...{edit_order}")
            original = 'old'
            target = 'new'

            file_name = f'{edit_order}.json'
            output_file = os.path.join(output_dir, file_name)

            if not os.path.exists(output_file):
                input_file_path = os.path.join(test_file_dir, file_name)
            else:
                input_file_path = output_file

            with open(input_file_path, 'r', encoding='utf-8') as fr:
                data_dict = json.load(fr)

            for data in data_dict['data']:

                if "model_output" in data:
                    print(f"第{data_dict['data'].index(data) + 1}条已经预测过，跳过该数据！")
                    continue

                print(f"正在预测第{data_dict['data'].index(data) + 1}条")

                # ... [处理每条数据的代码] ...
                dependency = data['dependency']
                old_version = data[f'{original}_version']
                new_version = data[f'{target}_version']
                original_version = data['dependency'] + data[f'{original}_version']
                target_version = data['dependency'] + data[f'{target}_version']
                original_code = data[f'{original}_code']
                masked_code = data[f'{target}_code']
                description = data['description']

                instruction = bulid_prompt(original_version, original_code, target_version, description)
                truncated_text = truncate_text(instruction, max_tokens)
                prediction = safe_predict(truncated_text,max_tokens, 6)
                data['model_output'] = str(prediction)

                # 定期保存结果
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

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
