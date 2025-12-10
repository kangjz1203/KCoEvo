import json
import os
import sys
import re


def escape_string(s):
    """转义字符串中的特殊字符"""
    if s is None:
        return ""
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')


class JsonToNQuadsConverter:
    def __init__(self, json_file_path, output_base_dir, prefix=""):
        self.json_file_path = json_file_path
        self.output_base_dir = output_base_dir
        self.prefix_base = prefix.rstrip('/')
        self.nquads = []  # This will be reset for each API

    def load_json_data(self):
        with open(self.json_file_path, "r", encoding="utf-8") as f:
            try:
                load_dict = json.load(f)
                return load_dict["data"]
            except Exception as e:
                print(f"Error loading or parsing JSON file: {self.json_file_path}")
                print(e)
                return None

    def _sanitize_for_uri_part(self, entity_name_part):
        return re.sub(r'[<>\"\'`#\s]', '_', entity_name_part)

    def _get_uri(self, entity_name_part, current_prefix):
        # Assumes entity_name_part is the final part to be appended after sanitization
        safe_entity_name = self._sanitize_for_uri_part(entity_name_part)
        return f"<{current_prefix}{safe_entity_name}>"

    def _add_quad_to_list(self, subject, predicate, obj, current_graph_name):  # Added nquads_list back
        # quad = f"{subject} {predicate} {obj} <{current_graph_name}> ."
        quad = f"{subject} {predicate} {obj}."
        self.nquads.append(quad)  # Appends to the instance's self.nquads

    def _api_exists_in_file(self, file_path, api_uri_str_unescaped_for_subject_check):
        if not os.path.exists(file_path):
            return False

        api_uri_as_subject = f"<{api_uri_str_unescaped_for_subject_check}>"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith(api_uri_as_subject + " "):
                        return True
        except IOError as e:
            print(f"Error reading file {file_path} for API existence check: {e}")
        return False

    def convert(self):
        libraries_data = self.load_json_data()
        if libraries_data is None:
            return

        for lib_name, versions_data in libraries_data.items():
            if not isinstance(versions_data, dict):
                print(f"Skipping library {lib_name} due to invalid versions_data format.")
                continue

            for version_str, apis_in_version in versions_data.items():
                if not isinstance(apis_in_version, dict):
                    print(f"Skipping version {version_str} for library {lib_name} due to invalid API data format.")
                    continue

                current_graph_name_suffix = f"{lib_name}/{version_str}"
                current_graph_name = f"{self.prefix_base}/{current_graph_name_suffix}"
                current_prefix = f"{self.prefix_base}/{current_graph_name_suffix}/"

                # Directory for this specific library and version (API-level files will go here)
                version_specific_output_path = os.path.join(self.output_base_dir, lib_name, version_str)
                if not os.path.exists(version_specific_output_path):
                    os.makedirs(version_specific_output_path, exist_ok=True)

                print(
                    f"Processing: Library={lib_name}, Version={version_str}, Output base: {version_specific_output_path}")

                for api_qualified_name, api_details in apis_in_version.items():
                    self.nquads = []  # Reset N-Quads list for EACH API

                    if api_details is None or not isinstance(api_details, dict):
                        print(f"Skipping invalid API data for {api_qualified_name} in {lib_name} {version_str}")
                        continue

                    entity_id = api_details.get("_id")
                    if not entity_id:
                        print(f"Skipping API entry with missing '_id': {api_details.get('name', 'Unknown API')}")
                        continue

                    # Define the specific file path for this API
                    sanitized_entity_id_for_filename = self._sanitize_for_uri_part(entity_id)  # Sanitize for filename
                    api_level_nquad_filename = f"{sanitized_entity_id_for_filename}.nq"
                    api_level_nquad_filepath = os.path.join(version_specific_output_path, api_level_nquad_filename)

                    # URI part for checking (without <>)
                    api_uri_str_for_check = f"{current_prefix}{sanitized_entity_id_for_filename}"

                    if self._api_exists_in_file(api_level_nquad_filepath, api_uri_str_for_check):
                        print(f"API {entity_id} already exists in {api_level_nquad_filepath}. Skipping.")
                        continue

                        # --- Proceed to generate N-Quads for this API as it's new or file doesn't exist ---
                    print(f"  Generating N-Quads for API: {entity_id}")
                    id_parts = entity_id.split(".")
                    package_name_from_id = id_parts[0]

                    module_name_from_details = api_details.get("module")
                    if module_name_from_details and entity_id.startswith(module_name_from_details):
                        module_name = module_name_from_details
                    elif len(id_parts) > 1:
                        module_name = ".".join(id_parts[:-1]) if api_details.get("type") != "module" else entity_id
                    else:
                        module_name = package_name_from_id

                    sanitized_package_name = self._sanitize_for_uri_part(package_name_from_id)
                    sanitized_module_name = self._sanitize_for_uri_part(module_name)
                    # api_uri_str_for_check is already sanitized entity_id part

                    package_uri = f"<{current_prefix}{sanitized_package_name}>"
                    module_uri = f"<{current_prefix}{sanitized_module_name}>"
                    api_uri = f"<{api_uri_str_for_check}>"

                    if api_details.get("is_deprecated", False):
                        self._add_quad_to_list(
                            api_uri, f"<{current_prefix}is_deprecated>", '"True"', current_graph_name
                        )

                    if sanitized_package_name != sanitized_module_name:
                        self._add_quad_to_list(
                            package_uri, f"<{current_prefix}has_module>", module_uri, current_graph_name
                        )

                    if sanitized_module_name != sanitized_entity_id_for_filename:  # Compare sanitized parts
                        self._add_quad_to_list(
                            module_uri, f"<{current_prefix}has_api>", api_uri, current_graph_name
                        )

                    api_type = api_details.get("type", "unknown")
                    self._add_quad_to_list(
                        api_uri, f"<{self.prefix_base}/rdf-schema#type>", f"<{current_prefix}{api_type}>",
                        current_graph_name
                    )

                    if api_type == "member_function" and api_details.get("class_context"):
                        class_context_uri_name_part = f"{module_name}.{api_details['class_context']}"
                        class_context_uri = self._get_uri(class_context_uri_name_part,
                                                          current_prefix)  # _get_uri sanitizes
                        self._add_quad_to_list(
                            api_uri, f"<{current_prefix}is_method_of>", class_context_uri, current_graph_name
                        )
                        self._add_quad_to_list(
                            class_context_uri, f"<{current_prefix}has_method>", api_uri, current_graph_name
                        )

                    parameters = api_details.get("parameters", {})
                    if parameters is not None and isinstance(parameters, dict):
                        for param_name, param_data in parameters.items():
                            if not isinstance(param_data, dict):
                                print(f"Skipping invalid parameter data for {param_name} in {api_qualified_name}")
                                continue

                            param_uri_name_part = f"{entity_id}/parameter/{param_name}"
                            param_uri = self._get_uri(param_uri_name_part, current_prefix)  # _get_uri sanitizes
                            self._add_quad_to_list(
                                api_uri, f"<{current_prefix}has_parameter>", param_uri, current_graph_name
                            )

                            if param_data.get("is_optional", False) is True:
                                self._add_quad_to_list(
                                    param_uri, f"<{current_prefix}is_optional>", '"True"', current_graph_name
                                )
                            else:
                                self._add_quad_to_list(
                                    param_uri, f"<{current_prefix}is_required>", '"True"', current_graph_name
                                )

                            # description = param_data.get("description")
                            # if description:
                            #     self._add_quad_to_list(
                            #         param_uri, f"<{current_prefix}has_description>", f'"{escape_string(description)}"',
                            #         current_graph_name
                            #     )

                            default_value = param_data.get("default")
                            if default_value is not None:
                                self._add_quad_to_list(
                                    param_uri, f"<{current_prefix}has_default_value>",
                                    f'"{escape_string(str(default_value))}"', current_graph_name
                                )

                            annotation = param_data.get("annotation")
                            if annotation and annotation != 'None':
                                self._add_quad_to_list(
                                    param_uri, f"<{current_prefix}has_annotation>", f'"{escape_string(annotation)}"',
                                    current_graph_name
                                )

                    # Save N-Quads for this API to its own file (overwrite if exists but API was not found inside)
                    if self.nquads:  # Only save if there are quads to save for this API
                        self.save_api_nquads_to_file(api_level_nquad_filepath, self.nquads)
                        self.nquads = []  # Clear for the next API, good practice

    def save_api_nquads_to_file(self, file_path, nquads_list):  # Renamed from append_nquads_to_file
        """将单个 API 的 N-Quads 列表保存到文件 (覆盖模式)"""
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):  # Should have been created in convert() already
            try:
                os.makedirs(file_dir, exist_ok=True)
            except OSError as e:
                print(f"Error creating directory {file_dir}: {e}")
                return

        try:
            # Open in write mode ('w') to overwrite if file exists but API was not found,
            # or to create if file doesn't exist.
            with open(file_path, "w", encoding="utf-8") as f:
                for quad in nquads_list:
                    f.write(quad + "\n")
            print(f"Saved N-Quads for API to {file_path}")
        except IOError as e:
            print(f"Error writing to file {file_path}: {e}")


if __name__ == "__main__":
    input_json_file = "vace_data.json"
    # Output directory for API-level N-Quad files, structured by library/version
    output_nquads_base_dir = "../kg_data/N_Quads/vace_output/api_granularity_checked"

    if not os.path.exists(input_json_file):
        print(f"Input JSON file not found: {input_json_file}")
        sys.exit(1)

    base_uri_prefix = ""

    print(f"Processing {input_json_file}...")
    converter = JsonToNQuadsConverter(input_json_file, output_nquads_base_dir, prefix=base_uri_prefix)
    try:
        converter.convert()
    except Exception as e:
        print(f"An error occurred during conversion for {input_json_file}: {e}")
        import traceback

        traceback.print_exc()

    print("Processing complete.")