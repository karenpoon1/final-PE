def get_pathname(split_config, model_config, additional=None):
    filename = f'split_{split_config}_model_{model_config}'
    if additional:
        filename = f'{filename}_{additional}'
    raw_pathname = f'results/raw/{filename}.pt'
    plot_pathname = f'results/plots/{filename}.png'
    return raw_pathname, plot_pathname
