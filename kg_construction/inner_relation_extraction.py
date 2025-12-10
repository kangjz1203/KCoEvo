import json
import os.path
import sys

prefix = "http://purl.org/codekg_reasoner/"
write_dir = "../kg_data/N_Quads/completion"
# input_file_dir = "../kg_data/parsed_data/json_data"
# input_file_dir = "../kg_data/parsed_data/from_transformers"
input_file_dir = "../kg_data/parsed_data/json_data/completion"


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

def write_json(output_file,data_dict):
    """

    :param output_file:
    :param data_dict: data,(dict)
    :return: None
    """
    with open(output_file,"w") as of:
        of.write(json.dumps(data_dict,indent=4))
    print(f"Write to {output_file}")


def escape_string(s):
    """转义字符串中的特殊字符"""
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')


class JsonToNQuadsConverter:
    # def __init__(self, json_data, prefix="http://purl.org/codekg_reasoner/", graph_name_suffix=None):
    def __init__(self, json_data, prefix="", graph_name_suffix=None):
        self.json_data = json_data
        self.prefix = prefix+graph_name_suffix+'/'
        self.graph_name = f"{prefix}{graph_name_suffix}"  # 图名称
        self.nquads = []

    def load_json(self):
        """加载 JSON 文件并返回数据"""
        with open(self.json_data, "r") as f:
            load_dict = json.load(f)
            try:
                load_data = load_dict["data"]
            except Exception as e:
                print(e)
                print(self.json_data)
        return load_data

    def _get_uri(self, entity_name):
        """生成实体的 URI"""
        return f"<{self.prefix}{entity_name}>"

    def _add_quad(self, subject, predicate, obj):
        """添加一个四元组到 N-Quads 列表，包含图名称"""
        quad = f"{subject} {predicate} {obj} <{self.graph_name}> ."
        self.nquads.append(quad)

    def convert(self):
        """将 JSON 数据转换为 N-Quads 格式，表示层级包含关系和参数关系"""
        data = self.load_json()


        # import ipdb
        # ipdb.set_trace()
        # version_data = data["transformers"]
        # version = str(data['transformers'].keys()).lstrip("dict_keys(['").rstrip("'])")
        # api_data = version_data[version]
        api_data = data

        for entity_name, entity_data in api_data.items():
            entity_id = entity_data["_id"]
            # 提取 package、module 和 function
            package_name = entity_id.split(".")[0]  # 假设 package 是第一个部分
            module_name = ".".join(entity_id.split(".")[:-1])  # 假设 module 是除最后一部分外的所有部分
            function_name = entity_id.split(".")[-1]


            # 生成 URI
            package_uri = self._get_uri(package_name)
            module_uri = self._get_uri(module_name)
            function_uri = self._get_uri(function_name)

            # 如果 function 是废弃的，替换为空 function
            if entity_data.get("is_deprecated", False):
                function_uri = self._get_uri("removed_function")  # 空 function 的 URI

            # 添加 package has module
            self._add_quad(package_uri, f"<{self.prefix}has_module>", module_uri)

            # 添加 module has function
            self._add_quad(module_uri, f"<{self.prefix}has_function>", function_uri)

            # 添加 function has parameter
            parameters = entity_data.get("parameters", {})
            for param_name, param_data in parameters.items():
                param_uri = self._get_uri(f"{function_name}/parameter/{param_name}")
                self._add_quad(function_uri, f"<{self.prefix}has_parameter>", param_uri)

                # 如果参数是可选的，添加标注
                if param_data.get("is_optional") is not False:
                    self._add_quad(param_uri, f"<{self.prefix}is_optional>", '"True"')
                else:
                    self._add_quad(param_uri, f"<{self.prefix}is_required>", '"True"')
                # 处理 description

                # 处理 description
                if param_data.get("description") is not None:
                    description_value = escape_string(param_data.get("description"))
                    self._add_quad(param_uri, f"<{self.prefix}has_description>", f'"{description_value}"')

                # 处理 default value
                if param_data.get("default") is not None:
                    default_value = param_data.get("default")
                    # 转义双引号和反斜杠
                    default_value = default_value.replace('"', '\\"').replace('\\', '\\\\')
                    self._add_quad(param_uri, f"<{self.prefix}has_default_value>", f'"{default_value}"')

        return self.nquads

    def save_to_file(self, file_dir,file_name):
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        """将 N-Quads 保存到文件"""
        file_path = os.path.join(file_dir,file_name+".nq")
        with open(file_path, "w", encoding="utf-8") as f:
            for quad in self.nquads:
                f.write(quad + "\n")


if __name__ == "__main__":
    # package_name = sys.argv[1]
    package_name = "packaging"
    input_path = os.path.join(input_file_dir,package_name)

    for file in os.listdir(input_path):

        if file.endswith(".json") and not file.endswith("change.json"):
            file_path = os.path.join(input_path, file)
            graph_name_suffix = package_name+file.split("/")[-1].replace(".json","")

            converter = JsonToNQuadsConverter(file_path,graph_name_suffix=graph_name_suffix)
            try:
                nquads = converter.convert()
            except Exception as e:
                print(e)
                print(file_path)
            file_dir = os.path.join(write_dir,package_name)
            converter.save_to_file(file_dir,graph_name_suffix)

