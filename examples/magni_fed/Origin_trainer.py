import math
import os
import time
import numpy as np

from plato.config import Config
from plato.trainers import basic

class Origin_trainer(basic.Trainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.client_quality = None
        self.epoch_rate = 1
        self.alpha = 1
        self.epoch_losses = []
        self.final_loss = 0.0

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
        self.epoch_losses = []  # Reset epoch losses for this training run
        self.final_loss = 0.0

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

            # Get and save the average loss for this epoch
            epoch_loss = self._loss_tracker.average
            self.run_history.update_metric("train_loss", epoch_loss)

            # Store loss in memory for reporting
            self.epoch_losses.append(epoch_loss)
            # Update final loss with the latest epoch loss
            self.final_loss = epoch_loss

            self.train_epoch_end(config)
            self.callback_handler.call_event("on_train_epoch_end", self, config)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)

    def save_magnified_parameters(self, filename):
        with open(filename, 'a') as f:
            f.write(f"{self.get_alpha()},{self.get_epoch_rate()},{os.getpid()}\n")

    def get_epoch_losses(self):
        """Return the list of loss values for all epochs."""
        return self.epoch_losses

    def get_final_loss(self):
        """Return the final loss value from training."""
        return self.final_loss