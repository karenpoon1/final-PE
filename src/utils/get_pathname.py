def get_pathname(split_config, model_config, additional=None):
    pathname = f'results/raw/split_{split_config}_model_{model_config}.pt'
    if additional:
        pathname = f'{pathname}_{additional}'
    return pathname
