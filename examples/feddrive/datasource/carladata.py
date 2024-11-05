from carla_dataset import CarlaMVDetDataset
from carla_loader import create_carla_loader
from dataset_factory import create_carla_dataset
import logging
import torch
import torch.nn as nn
from functools import partial

from timm.data import create_carla_dataset, create_carla_loader
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    load_checkpoint,
    convert_splitbn_model,
    model_parameters,
)
from timm.utils import *
from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler


#from plato.clients import simple
from plato.datasources import base
from plato.trainers import basic

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
    "train_towns" : [1,2,3,4,5,6,7,10],
    "train_weathers" : [0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19],
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
    
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        freeze_num=args.freeze_num,
    )
    
    # setup loss function
    if args.smoothing > 0:
        cls_loss = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        cls_loss = nn.CrossEntropyLoss()

    if args.smoothed_l1:
        l1_loss = torch.nn.SmoothL1Loss
    else:
        l1_loss = torch.nn.L1Loss
    
    train_loss_fns = {
        #"traffic": MVTL1Loss(1.0, l1_loss=l1_loss),
        "traffic": LAVLoss(),
        "waypoints": torch.nn.L1Loss(),
        "cls": cls_loss,
        "stop_cls": cls_loss,
    }
    validate_loss_fns = {
        #"traffic": MVTL1Loss(1.0, l1_loss=l1_loss),
        "traffic": LAVLoss(),
        "waypoints": torch.nn.L1Loss(),
        "cls": cls_loss,
        "stop_cls": cls_loss,
    }
    
    for examples, labels in loader:
        print(examples)
        break