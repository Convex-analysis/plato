import asyncio
import logging
import math
import random
import time
from cnn import CNN_OriginalFedAvg
import numpy as np

from Origin_trainer import Origin_trainer
from Origin_client import Origin_client
from Origin_server import Origin_server
from MaroScheduleServer import Macro_server  # Add this line to import Macro_server
from plato.config import Config



#不聚合不合格的client
class Scheme1_server(Origin_server):
    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)
        qualified_client_count = 0
        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples
            if report.quality:
                logging.info("This client is qualified for aggregation.")
                qualified_client_count += 1
                for name, delta in update.items():
                    # Use weighted average by the number of samples
                    avg_update[name] += delta * (num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)
        if qualified_client_count == 0:
            logging.info("No client is qualified for aggregation.")
        return avg_update
    
    def choose_clients(self, clients_pool, clients_count):
        """Chooses a subset of the clients to participate in each round."""
        #clients_count = math.floor(self.current_round / 10) + clients_count
        if self.current_round % 20 == 0:
            clients_count = clients_count + 1
        if clients_count >= len(clients_pool):
            clients_count = len(clients_pool)
        self.clients_per_round = clients_count
        random.setstate(self.prng_state)
        # Select clients randomly
        
        selected_clients = random.sample(clients_pool, clients_count)

        self.prng_state = random.getstate()
        logging.info("[%s] Selected clients: %s", self, selected_clients)
        return selected_clients
#Origin_server就是第三中我们自己的聚合方法

def main():
    with_MRscehduler = True
    #使用两层的cnn
    model = CNN_OriginalFedAvg
    trainer = Origin_trainer
    client = Origin_client(trainer=trainer)
    
    server = Scheme1_server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
