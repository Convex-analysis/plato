## LMdrive里的data相关的文件
- carla_dataset.py
- carla_loader.py
两个都在vision_encoder/timm/data里
* 调用逻辑(全部基于vision_encoder)：
从train_pretrain.py中调用了./timm/data下的dataset_factory.py里的create_carla_dataset()函数，这个函数会调用./timm/data/carla_dataset.py里的CarlaMVDetDataset类
** CarlaMVDetDatase类：
  这个类里包含了数据的处理，并以目录的形式存储了数据，还有一个_extract_data_item（）函数用来根据目录提取数据
之后在train_pretrain.py的1158行，加载的数据被load进create_carla_loader()位于./timm/data/carla_loader.py
** 里面的大部分处理功能都是vision_encoder\timm\data\transforms_carla_factory.py中的函数负责的
sampler这里用了DDP sampler，下面是相关网址
- https://zhuanlan.zhihu.com/p/660485845
- https://pytorch.org/docs/stable/data.html

