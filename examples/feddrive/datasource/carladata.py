from carla_dataset import CarlaMVDetDataset
from carla_loader import create_carla_loader
from dataset_factory import create_carla_dataset
import logging
import torch
import torch.nn as nn
from functools import partial

#from plato.clients import simple
from plato.datasources import base
#from plato.trainers import basic

args = {
    "dataset" : "carla",
    "batch_size" : 24,
    "color_jitter" : 0.4,
    "data_dir" : "D:\EXP\CarlaData",
    "train_split" : "train",
    "epoch_repeats" : 0,
    "multi_view_input_size" : [3,128,128],
    "scale" : [0.9,1.1],
    "workers" : 4,
    "distributed" : False,
    "pin_mem" : False,
    "train_towns" : "Town01",
    "train_weathers" : 10,
    "with_lidar" : True,
    "with_seg" : False,
    "with_depth" : False,
    "multi_view" : True,
    "augment_prob" : 0.0,
    "temporal_frames" : False
}

data_config = {
    "input_size" : [3,224,224],
    "mean" : [0.485, 0.456, 0.406],
    "std" : [0.229, 0.224, 0.225]
}

train_interpolation = "bilinear"
collate_fn = None
path = 'D:\EXP\CarlaData'




class DataSource(base.DataSource):
    """A custom datasource with custom training and validation datasets."""

    def __init__(self):
        super().__init__()

        self.trainset = create_carla_dataset(
            args["dataset"],
            root=args["data_dir"],
            towns=args["train_towns"],
            weathers=args["train_weathers"],
            batch_size=args["batch_size"],
            with_lidar=args["with_lidar"],
            with_seg=args["with_seg"],
            with_depth=args["with_depth"],
            multi_view=args["multi_view"],
            augment_prob=args["augment_prob"],
            temporal_frames=args["temporal_frames"],
        )


def initialize_loader():
        carlaSource = DataSource()
        data_train_loader = create_carla_loader(
        carlaSource.trainset,
        input_size=data_config["input_size"],
        batch_size=args["batch_size"],
        multi_view_input_size=args["multi_view_input_size"],
        is_training=True,
        scale=args["scale"],
        color_jitter=args["color_jitter"],
        interpolation=train_interpolation,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args["workers"],
        distributed=args["distributed"],
        collate_fn=collate_fn,
        pin_memory=args["pin_mem"],
    )
        return data_train_loader
    
if __name__ == "__main__":
    
    loader = initialize_loader()   
    
    for i, (data, target) in enumerate(loader):
        print(i)