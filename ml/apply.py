from typing import List
import re
import yaml

def read_extract_config(config_path):
    number_with_unit_pattern = re.compile(r'^\s*(-?\d+(\.\d+)?)(.*)$')

    with open(config_path, 'r') as f:
        config =  yaml.safe_load(f)

    extracted_data = []

    def traverse(node, path=''):
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = f"{path}.{key}" if path else key
                traverse(value, new_path)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                new_path = f"{path}[{index}]"
                traverse(item, new_path)
        elif isinstance(node, str):
            match = number_with_unit_pattern.match(node)
            if match:
                number = float(match.group(1))
                unit = match.group(3).strip()
                full_path = f"{path}|{unit}" if unit else path  # Append unit if available
                extracted_data.append({'path': full_path, 'value': number})
        elif isinstance(node, (int, float)):
            extracted_data.append({'path': path, 'value': node})

    traverse(config)
    return extracted_data

controllable_params: List[str] = []

# add numeric params to global list
def read_configs(cfgs: List[str]):
    for cfg in cfgs:
        extracted_data = read_extract_config(cfg)
        controllable_params.extend(extracted_data)

def apply(features: List[str]):
    pass

# for feature importance
def add_context(params: List[str]):
    pass

# for model
def get_context() -> List[str]:
    pass