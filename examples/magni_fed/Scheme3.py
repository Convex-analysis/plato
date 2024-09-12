import asyncio
import logging
import math
import time

import numpy as np

from Origin_trainer import Origin_trainer
from Origin_client import Origin_client
from Origin_server import Origin_server
from plato.config import Config


def main():
    trainer = Origin_trainer
    client = Origin_client(trainer=trainer)
    server = Origin_server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
