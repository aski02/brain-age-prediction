from .data_loaders.Charite_Dataset import Charite_Dataset
from .data_loaders.OpenBHB_Dataset import OpenBHB_Dataset


def dataset_loader(**configs):
    dataset_name = configs.pop("dataset_name")
    dataset_wrapper = {
        "Charite": Charite_Dataset,
        "OpenBHB": OpenBHB_Dataset,
    }
    dataset = dataset_wrapper[dataset_name](**configs)

    return dataset
