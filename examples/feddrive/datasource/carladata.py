from carla_dataset import CarlaMVDetDataset
from carla_loader import create_carla_loader
from dataset_factory import create_dataset
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
}

data_config = {
    "input_size" : [3,224,224],
    "mean" : [0.485, 0.456, 0.406],
    "std" : [0.229, 0.224, 0.225]
}
train_interpolation = "bilinear"
collate_fn = None
path = 'D:\EXP\CarlaData'
dataset_train = create_dataset(
            args["dataset"],
            root=args["data_dir"],
            split=args["train_split"],
            is_training=True,
            batch_size=args["batch_size"],
            repeats=args["epoch_repeats"],
        )

print(dataset_train.__len__())

data_train = create_carla_loader(
        dataset_train,
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

print(data_train.pin_memory)