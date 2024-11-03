## 修改的地方
- 所有的tools里data parsing的几个文件的写法，存在变量的跨域引用的问题 如 dataset_root，cur_dir
- carla_dataset.py里的line 180 要求读取的文件是dataset_index_test.txt，感觉应该是dataset_index.txt，故做了修改
- carla_dataset.py里的line 187 构建的数据集结构和LMdrive里的readme不一样，原码加入了一个中间文件夹data存放所有数据，但是readme里dataroot下就全是数据。故作修改