from plato.datasources import base
from timm.data import create_carla_dataset, create_carla_loader

class DataSource(base.DataSource):
    """A custom datasource with custom training and validation datasets."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.trainset = create_carla_dataset(
            self.args.dataset, # newcarla
            root=self.args.data_dir, # DATASET_ROOT='/path/to/your/dataset'
            towns=self.args.train_towns, # 1 2 3 4 5 6 7 10
            weathers=self.args.train_weathers, # 0 1 2 3 4 5 6 7 8 9 10 11 14 15 16 17 18 19
            batch_size=self.args.batch_size, # 24
            with_lidar=self.args.with_lidar,
            with_seg=self.args.with_seg,
            with_depth=self.args.with_depth,
            multi_view=self.args.multi_view,
            augment_prob=self.args.augment_prob,
            temporal_frames=self.args.temporal_frames,
        )

        self.testset = create_carla_dataset(
            self.args.dataset, # newcarla
            root=self.args.data_dir, # DATASET_ROOT='/path/to/your/dataset'
            towns=self.args.val_towns, # 1 5 7
            weathers=self.args.val_weathers, # 12 13 20
            batch_size=self.args.batch_size, # 24
            with_lidar=self.args.with_lidar,
            with_seg=self.args.with_seg,
            with_depth=self.args.with_depth,
            multi_view=self.args.multi_view,
            augment_prob=self.args.augment_prob,
            temporal_frames=self.args.temporal_frames,
        )

