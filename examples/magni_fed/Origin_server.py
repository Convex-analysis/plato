import asyncio
import logging

from plato.servers import fedavg

class Origin_server(fedavg.Server):
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