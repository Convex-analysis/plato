"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

import logging

from datasets import mnist, fashion_mnist, cifar10
from config import Config

registered_datasets = {
    'MNIST': mnist,
    'FashionMNIST': fashion_mnist,
    'CIFAR10': cifar10
}


def get():
    """Get the dataset with the provided name."""
    dataset_name = Config().training.dataset
    data_path = Config().training.data_path

    logging.info('Dataset: %s', Config().training.dataset)
    logging.info('Dataset path: %s', data_path)

    if dataset_name in registered_datasets:
        dataset = registered_datasets[dataset_name].Dataset(data_path)
    else:
        raise ValueError('No such dataset: {}'.format(dataset_name))

    return dataset