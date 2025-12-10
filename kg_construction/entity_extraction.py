import json
import os

parsed_data_dir = "../kg_data/parsed_data/completion"
# parsed_data_dir = "../kg_data/parsed_data/original_data"
output_dir = "../kg_data/parsed_data/json_data/completion"

def load_json(input_file):
    """加载 JSON 文件并返回数据"""
    with open(input_file, "r") as f:
        load_dict = json.load(f)
        try:
            load_data = load_dict["data"]
        except Exception as e:
            print("Error:{e}\n\n")
            print(input_file)
    return load_data

def write_json(output_file, data_dict):
    """将字典数据写入 JSON 文件"""
    with open(output_file, "w") as of:
        of.write(json.dumps(data_dict, indent=4))

if __name__ == '__main__':
    # 遍历 parsed_data_dir 目录
    for root, dirs, files in os.walk(parsed_data_dir):
        for file_name in files:
            # 检查文件是否以 .json 结尾且不以 _ 开头
            if file_name.endswith(".json") and not file_name.startswith("_"):
                file_path = os.path.join(root, file_name)

                load_data = load_json(file_path)


                # 遍历 load_data 的键值对
                for name, entries in load_data.items():
                    # 创建以 name 命名的子目录
                    name_dir = os.path.join(output_dir, name)
                    if not os.path.exists(name_dir):
                        os.makedirs(name_dir)

                    # 遍历 entries 的键值对
                    for version, data in entries.items():
                        # import ipdb
                        # ipdb.set_trace()
                        # 创建以 version 命名的文件路径

                        # api_name = list(data.keys())[0]
                        # try:
                        #     data[api_name]["version"] = name+version
                        # except:
                        #     print(file_name+version)
                        # data.update({"version":version})
                        write_path = os.path.join(name_dir, f"{version}.json")
                        write_dic = {"data":data}
                        write_json(write_path, write_dic)
                        print(f"Written: {write_path}")