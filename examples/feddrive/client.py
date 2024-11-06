import logging
import os
import time
import pickle
import sys
import copy
import numpy as np

from plato.clients import simple
from plato.config import Config
from plato.utils import fonts
from types import SimpleNamespace

class Client(simple.Client):
    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        """
        self.custom_model = model # custom model is used
        self.custom_datasource = datasource # custom datasource is used
        self.custom_trainer = trainer # custom trainer is used
        self.sampler = None
        self.testset_sampler = None  # Sampler for the test set
        """

    def configure(self) -> None:
        """Prepares this client for training."""
        # set model
        if self.model is None and self.custom_model is not None:
            self.model = self.custom_model

        # set trainer
        if self.trainer is None and self.custom_trainer is None:
            self.trainer = trainers_registry.get(
                model=self.model, callbacks=self.trainer_callbacks)
        elif self.trainer is None and self.custom_trainer is not None:
            self.trainer = self.custom_trainer(
                model=self.model, callbacks=self.trainer_callbacks)
        self.trainer.set_client_id(self.client_id)

        # set algorithm
        if self.algorithm is None and self.custom_algorithm is None:
            self.algorithm = algorithms_registry.get(trainer=self.trainer)
        elif self.algorithm is None and self.custom_algorithm is not None:
            self.algorithm = self.custom_algorithm(trainer=self.trainer)
        self.algorithm.set_client_id(self.client_id)

        # Pass inbound and outbound data payloads through processors for
        # additional data processing 在接收模型之后，以及更新完发送模型之前对模型的操作
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Client", client_id=self.client_id, trainer=self.trainer
        )

        # Setting the data sampler as None

    def _load_data(self) -> None:
        """Generates data and loads them onto this client."""
        # The only case where Config().data.reload_data is set to true is
        # when clients with different client IDs need to load from different datasets,
        # such as in the pre-partitioned Federated EMNIST dataset. We do not support
        # reloading data from a custom datasource at this time.
        # 如果数据集为空，则加载数据集
        if (
            self.datasource is None
            or (hasattr(Config().data, "reload_data") and Config().data.reload_data)
        ):
            logging.info("[%s] Loading its data source.", self)
            if self.custom_datasource is None:
                self.datasource = datasources_registry.get(client_id=self.client_id)
                # logging.info("[%s] client id: %d, custom_datasource is none", self, self.client_id)
            elif self.custom_datasource is not None:
                self.datasource = self.custom_datasource()
            logging.info(
                "[%s] Dataset size: %s", self, self.datasource.num_train_examples()
            )

        # if the model is not tested, client.do_test=False
        # if hasattr(Config().clients, "do_test") and Config().clients.do_test:
