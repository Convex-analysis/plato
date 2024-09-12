import logging
import random
from types import SimpleNamespace
import numpy as np

from plato.config import Config
from plato.clients import simple



class Origin_client(simple.Client):
    def __init__(
            self,
            model=None,
            datasource=None,
            algorithm=None,
            trainer=None,
            callbacks=None,
            trainer_callbacks=None,
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks, trainer_callbacks)
        self.quality = None  # Whether the client is qualified to complete the training and communication
        self.communication_time = 0
        self.training_time = 0
        self.sojourn_time = 0
        self.para_file = None

    def create_magnified_file(self):
        """当配置了max_concurrency时,会创建另一个线程的trainer，导致无法存储更改，因此创建创建文件存储"""
        # 在当前目录创建文件，文件名为self.client_id_magnified.csv
        filename = f"{self.client_id}_magnified.csv"
        with open(filename, 'w') as f:
            f.write("alpha,beta,process_id\n")
        # 保存文件名
        return filename

    def configure(self) -> None:
        """Prepares this client for training."""
        super().configure()
        logging.info("Configuration started")
        # num_train = self.sampler.num_samples()
        # model_size = self.get_model_size()
        # 根据config中的average_duration为基准，根据正态分布随机生成一个sojourn_time
        unqualified_ratio = 0
        average_duration = 0
        stand_devation = 3

        # 判断是否有max_concurrency参数，如果有则创建文件
        if hasattr(Config().trainer, "max_concurrency"):
            self.para_file = self.create_magnified_file()

        if hasattr(Config().parameters, "unqualified_ratio"):
            unqualified_ratio = Config().parameters.unqualified_ratio
        if hasattr(Config().parameters, "average_duration"):
            average_duration = Config().parameters.average_duration
        self.sojourn_time = random.normalvariate(average_duration, stand_devation)
        sojourn_time = self.sojourn_time

        # 根据config中的unqualified_ratio为0到1间的概率，将self.sojourn_time方法随机生成一个0.5倍到1.5倍的值
        if np.random.uniform(0, 1) < unqualified_ratio:
            sojourn_time = self.sojourn_time * random.uniform(1, 1.2)
            self.quality = False
        else:
            self.quality = True
        self.trainer.set_client_quality(self.quality)
        # 在0,1之间根据高斯分布采样，根据model_size得到一个0到1之间的值，作为communication_time的比例
        communication_rate = random.normalvariate(0.5, 0.15)
        training_rate = random.normalvariate(0.5, 0.1)
        total_rate = communication_rate + training_rate
        communication_rate = communication_rate / total_rate
        training_rate = training_rate / total_rate
        self.communication_time = sojourn_time * communication_rate
        self.training_time = sojourn_time * training_rate
        logging.info("sojourn_time: %f, communication_time: %f, training_time: %f" % (
            self.sojourn_time, self.communication_time, self.training_time))
        logging.info("Configuration ended")

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Wrap up generating the report with any additional information."""
        if hasattr(Config().trainer, "max_concurrency"):
            #读文件，获取alpha,beta,process_id
            with open(self.para_file, 'r') as f:
                lines = f.readlines()
                last_line = lines[-1]
                #判断是否有记录，若无则跳过
                if last_line == "alpha,beta,process_id\n":
                    return report
                alpha, beta, process_id = last_line.split(',')
                alpha = float(alpha)
                beta = float(beta)
                process_id = int(process_id)
                self.trainer.set_alpha(alpha)
                self.trainer.set_epoch_rate(beta)
        report.quality = self.quality
        report.epoch_rate = self.trainer.get_epoch_rate()
        report.alpha = self.trainer.get_alpha()

        #记录训练数据后将对应参数回归初始化设置
        self.trainer.set_epoch_rate(1)
        self.trainer.set_alpha(1)
        logging.info("quality: %s, epoch_rate: %f, alpha: %f" % (report.quality, report.epoch_rate, report.alpha))
        return report