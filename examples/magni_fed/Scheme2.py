import asyncio
import logging
import math
import random
import time

import numpy as np

from Origin_trainer import Origin_trainer
from Origin_client import Origin_client
from Origin_server import Origin_server
from plato.config import Config
from MaroScheduleServer import Macro_server # Add this line to import Macro_server

#FDP-FL
#按执行的比例聚合
class Scheme2_server(Origin_server):
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
            excuted_rate = report.epoch_rate
            num_samples = report.num_samples
            if report.quality:
                logging.info("This client is qualified for aggregation.")

                for name, delta in update.items():
                    # Use weighted average by the number of samples
                    avg_update[name] += delta * (excuted_rate * num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)
        return avg_update
#Origin_server就是第三中我们自己的聚合方法

def main():
    with_MRscehduler = True
    trainer = Origin_trainer
    client = Origin_client(trainer=trainer)

    server = Scheme2_server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
