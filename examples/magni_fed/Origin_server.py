import asyncio
import logging
import statistics

from plato.servers import fedavg
from plato.config import Config

class Origin_server(fedavg.Server):
    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.current_round_losses = []
        self.avg_loss = 0.0

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)
        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        # Reset the list of losses for this round
        self.current_round_losses = []

        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples
            magnified_factor = report.alpha
            logging.info("the corresponding magnified ratio is %f" % magnified_factor)

            # Collect loss information if available
            if hasattr(report, 'final_loss'):
                client_id = updates[i].client_id
                loss = report.final_loss

                # Store loss for this client
                self.current_round_losses.append(loss)

                logging.info(f"[Server] Client {client_id} reported loss: {loss}")

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (magnified_factor * num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        # Calculate average loss for this round if we have any losses
        if self.current_round_losses:
            self.avg_loss = statistics.mean(self.current_round_losses)
            logging.info(f"[Server] Average loss for round {self.current_round}: {self.avg_loss}")

        return avg_update

    def get_logged_items(self) -> dict:
        """Get items to be logged by the LogProgressCallback class in a .csv file."""
        logged_items = super().get_logged_items()

        # Add the average loss to the logged items
        logged_items["loss"] = self.avg_loss

        return logged_items