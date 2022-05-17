import torch

from utils.get_pathname import get_pathname

def load_result(split_config, model_config, additional):
    raw_pathname, _ = get_pathname(split_config, model_config, additional)
    results = torch.load(raw_pathname)
    return results
