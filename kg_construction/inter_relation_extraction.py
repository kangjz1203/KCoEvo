import json
import os.path
import sys

"""

python inter_relation_extraction.py imageio 1.5 1.6

"""


prefix = "http://purl.org/codekg_reasoner/"
write_dir = "../kg_data/N_Quads"
input_file_dir = "../kg_data/parsed_data/json_data"


def load_json(input_file):
    """加载 JSON 文件并返回数据"""
    with open(input_file, "r") as f:
        load_dict = json.load(f)
        try:
            load_data = load_dict["data"]
        except Exception as e:
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


class JsonToNQuadsConverter:
    def __init__(self, json_data, prefix="http://purl.org/codekg_reasoner/", package=None,old_version=None,new_version=None):
        self.json_data = json_data
        self.prefix = prefix
        self.package = package
        self.old_prefix = prefix + package + old_version + '/'
        self.new_prefix = prefix + package + new_version + '/'
        # self.change_prefix = prefix + package + old_version+'_'+new_version+'_'+'/'
        self.graph_name = f"{prefix}{package}"  # 图名称
        self.old_version = old_version
        self.new_version = new_version
        # self.nquads_old = []  # 旧版本的 N-Quads
        # self.nquads_new = []  # 新版本的 N-Quads
        self.nquads_intermediate = []  # 中间第三方的 N-Quads

    def load_json(self):
        """加载 JSON 文件并返回数据"""
        with open(self.json_data, "r") as f:
            load_dict = json.load(f)
            try:
                load_data = load_dict["data"]
            except Exception as e:
                print(self.json_data)
        return load_data

    def _get_uri(self, entity_name,package,version=None):
        """生成实体的 URI"""
        return f"<{self.prefix}{package}{version}/{entity_name}>"

    def _add_quad(self, subject, predicate, obj, nquads_list):
        """添加一个四元组到指定的 N-Quads 列表，包含图名称"""
        quad = f"{subject} {predicate} {obj} <{self.graph_name}> ."
        nquads_list.append(quad)

    def _handle_addition(self, entity_data, nquads_list,prefix):
        """处理添加类的 pattern"""
        entity_id = entity_data["_id"].split(".")[-1]
        # entity_id = entity_data["name"]
        entity_uri = self._get_uri(entity_id,self.package,self.new_version)
        blank_node = self._get_uri("blank_node",self.package,self.old_version)
        self._add_quad(blank_node, f"<{prefix}/has_added_to>", entity_uri, nquads_list)

    def _handle_deprecation(self, entity_data, nquads_list,prefix):
        """处理删除类的 pattern"""
        entity_id = entity_data["_id"].split(".")[-1]
        # entity_id = entity_data["name"]
        entity_uri = self._get_uri(entity_id,self.package,self.old_version)
        blank_node = self._get_uri("blank_node",self.package,self.new_version)
        self._add_quad(entity_uri, f"<{prefix}/has_removed_to>", blank_node, nquads_list)

    # def _handle_parameter_reordering(self, entity_data, nquads_list,prefix):
    #     """处理参数重排序的 pattern"""
    #     entity_id = entity_data["_id"]
    #     entity_uri = self._get_uri(entity_id,prefix)
    #     reordering_changes = entity_data.get("reordering_changes", {})
    #     old_order = reordering_changes.get("old_order", [])
    #     new_order = reordering_changes.get("new_order", [])
    #
    #     # 添加旧参数顺序
    #     self._add_quad(entity_uri, f"<{prefix}has_old_parameter_order>", f'"{", ".join(old_order)}"', nquads_list)
    #     # 添加新参数顺序
    #     self._add_quad(entity_uri, f"<{prefix}has_new_parameter_order>", f'"{", ".join(new_order)}"', nquads_list)
    #
    # def _handle_parameter_default_value_change(self, entity_data, nquads_list,prefix):
    #     """处理参数默认值变化的 pattern"""
    #     entity_id = entity_data["_id"]
    #     entity_uri = self._get_uri(entity_id,prefix)
    #     default_value_changes = entity_data.get("default_value_changed", [])
    #
    #     for change in default_value_changes:
    #         param_name = change["parameter"]
    #         param_uri = self._get_uri(f"{entity_id}/parameter/{param_name}",prefix)
    #         old_default = change["old_default"]
    #         new_default = change["new_default"]
    #
    #         # 添加旧默认值
    #         self._add_quad(param_uri, f"<{prefix}has_old_default_value>", f'"{old_default}"', nquads_list)
    #         # 添加新默认值
    #         self._add_quad(param_uri, f"<{prefix}has_new_default_value>", f'"{new_default}"', nquads_list)

    def convert(self):
        """将 JSON 数据转换为 N-Quads 格式，表示层级包含关系和参数关系"""
        data = self.load_json()
        # import ipdb
        # ipdb.set_trace()
        for entity_data in data:
            pattern = entity_data.get("pattern")
            prefix_package = self.prefix + self.package
            if pattern:
                if pattern.endswith("_addition"):
                    # self._handle_addition(entity_data, self.nquads_new,self.new_prefix)

                    self._handle_addition(entity_data, self.nquads_intermediate,prefix_package)
                elif pattern.endswith("_removal"):
                    # self._handle_deprecation(entity_data, self.nquads_old,self.old_prefix)
                    self._handle_deprecation(entity_data, self.nquads_intermediate,prefix_package)
                # elif pattern == "parameter_reordering":
                #     # 处理参数重排序
                #     self._handle_parameter_reordering(entity_data, self.nquads_intermediate,self.change_prefix)
                # elif pattern == "parameter_default_value_change":
                #     # 处理参数默认值变化
                #     self._handle_parameter_default_value_change(entity_data, self.nquads_intermediate,self.change_prefix)
                # 其他 pattern 的处理逻辑可以在这里添加

        # return self.nquads_old, self.nquads_new, self.nquads_intermediate
        return self.nquads_intermediate

    def save_to_file(self, file_dir, package_name):
        # package_path = os.path.join(file_dir,package_name)
        # if not os.path.exists(package_path):
        #     os.makedirs(package_path)
        """将 N-Quads 保存到文件"""
        old_file_path = os.path.join(file_dir, f"{self.old_version}.nq")
        new_file_path = os.path.join(file_dir, f"{self.new_version}.nq")
        intermediate_dir_path = os.path.join(file_dir,"changes")
        if not os.path.exists(intermediate_dir_path):
            os.mkdir(intermediate_dir_path)
        intermediate_file_path = os.path.join(intermediate_dir_path, f"{self.old_version}_{self.new_version}_changes.nq")

        # with open(old_file_path, "a", encoding="utf-8") as f:
        #     for quad in self.nquads_old:
        #         f.write(quad + "\n")
        #
        # with open(new_file_path, "a", encoding="utf-8") as f:
        #     for quad in self.nquads_new:
        #         f.write(quad + "\n")
        # import ipdb
        # ipdb.set_trace()
        with open(intermediate_file_path, "w", encoding="utf-8") as f:
            for quad in self.nquads_intermediate:
                f.write(quad + "\n")


if __name__ == "__main__":
    package_name = sys.argv[1]
    _old_version = sys.argv[2]
    _new_version = sys.argv[3]

    old_version = "=="+_old_version
    new_version = "=="+_new_version

    filename = old_version+"_"+new_version+"_"+"changes"+".json"
    input_path = os.path.join(input_file_dir,package_name,"changes", filename)
    graph_name_suffix = package_name
    converter = JsonToNQuadsConverter(input_path, package=package_name,old_version=old_version,new_version=new_version)
    try:
        # nquads_old, nquads_new, nquads_intermediate = converter.convert()
        nquads_intermediate = converter.convert()
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
    file_dir = os.path.join(write_dir, package_name)
    converter.save_to_file(file_dir, graph_name_suffix)