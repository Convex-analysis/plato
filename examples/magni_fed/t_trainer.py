import asyncio
import logging
import math
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from plato.config import Config
from plato.servers import fedavg

from plato.clients import simple
from plato.trainers import basic


# a test trainer
class T_trainer(basic.Trainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.client_quality = None
        self.epoch_rate = 1
        self.alpha = 1

    def set_client_quality(self, quality):
        self.client_quality = quality

    def set_epoch_rate(self, epoch_rate):
        self.epoch_rate = epoch_rate

    def get_epoch_rate(self):
        return self.epoch_rate

    def set_alpha(self, alpha):
        self.alpha = alpha

    def get_alpha(self):
        return self.alpha

    def train_model(self, config, trainset, sampler, **kwargs):
        """The default training loop when a custom training loop is not supplied."""
        batch_size = config["batch_size"]
        self.sampler = sampler
        tic = time.perf_counter()

        self.run_history.reset()

        self.train_run_start(config)
        self.callback_handler.call_event("on_train_run_start", self, config)

        self.train_loader = self.get_train_loader(batch_size, trainset, sampler)

        # Initializing the loss criterion
        self._loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        self.optimizer = self.get_optimizer(self.model)
        self.lr_scheduler = self.get_lr_scheduler(config, self.optimizer)
        self.optimizer = self._adjust_lr(config, self.lr_scheduler, self.optimizer)

        self.model.to(self.device)
        self.model.train()

        total_epochs = config["epochs"]

        # 随机生成（0.5到1）之间的数字，作为total_epochs的缩小比例
        if not self.client_quality:
            total_epochs_rate = np.random.uniform(0.5, 0.7)
            # 向上取整
            total_epochs = math.ceil(total_epochs * total_epochs_rate)
            self.set_epoch_rate((1 / total_epochs_rate))
            self.set_alpha(np.random.uniform(1, self.get_epoch_rate()))

            if "max_concurrency" in config:
                # 若存在文件"self.client_id_magnified.csv",则存储alpha,beta,process_id
                filename = f"{self.client_id}_magnified.csv"
                self.save_magnified_parameters(filename)

        for self.current_epoch in range(1, total_epochs + 1):
            self._loss_tracker.reset()
            self.train_epoch_start(config)
            self.callback_handler.call_event("on_train_epoch_start", self, config)

            for batch_id, (examples, labels) in enumerate(self.train_loader):
                self.train_step_start(config, batch=batch_id)
                self.callback_handler.call_event(
                    "on_train_step_start", self, config, batch=batch_id
                )

                examples, labels = examples.to(self.device), labels.to(self.device)

                loss = self.perform_forward_and_backward_passes(
                    config, examples, labels
                )

                self.train_step_end(config, batch=batch_id, loss=loss)
                self.callback_handler.call_event(
                    "on_train_step_end", self, config, batch=batch_id, loss=loss
                )

            self.lr_scheduler_step()

            if hasattr(self.optimizer, "params_state_update"):
                self.optimizer.params_state_update()

            # Simulate client's speed
            if (
                    self.client_id != 0
                    and hasattr(Config().clients, "speed_simulation")
                    and Config().clients.speed_simulation
            ):
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if (
                    hasattr(Config().server, "request_update")
                    and Config().server.request_update
            ):
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{self.current_epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)

            self.run_history.update_metric("train_loss", self._loss_tracker.average)
            self.train_epoch_end(config)
            self.callback_handler.call_event("on_train_epoch_end", self, config)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)

    def save_magnified_parameters(self, filename):
        with open(filename, 'a') as f:
            f.write(f"{self.get_alpha()},{self.get_epoch_rate()},{os.getpid()}\n")


class custom_cap_client(simple.Client):
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
                alpha, beta, process_id = last_line.split(',')
                alpha = float(alpha)
                beta = float(beta)
                process_id = int(process_id)
                self.trainer.set_alpha(alpha)
                self.trainer.set_epoch_rate(beta)
        report.quality = self.quality
        report.epoch_rate = self.trainer.get_epoch_rate()
        report.alpha = self.trainer.get_alpha()
        logging.info("quality: %s, epoch_rate: %f, alpha: %f" % (report.quality, report.epoch_rate, report.alpha))
        return report


# a customized FL server with magnified aggregation
class T_server(fedavg.Server):
    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples
            magnified_factor = report.alpha
            logging.info("the corresponding magnified ratio is %f" % magnified_factor)
            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (magnified_factor * num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update


def main():
    trainer = T_trainer
    client = custom_cap_client(trainer=trainer)
    server = T_server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
