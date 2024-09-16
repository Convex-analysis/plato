import asyncio
import logging
import random

from plato.servers import fedavg
import math

from Origin_trainer import Origin_trainer
from Origin_client import Origin_client
from Origin_server import Origin_server
from plato.config import Config

class Macro_server(fedavg.Server):
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
    
    def choose_clients(self, clients_pool, clients_count):
        """Chooses a subset of the clients to participate in each round."""
        #clients_count = math.floor(self.current_round / 10) + clients_count
        if self.current_round % 10 == 0:
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

def main():
    trainer = Origin_trainer
    client = Origin_client(trainer=trainer)
    server = Macro_server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
