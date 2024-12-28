import os
import sys
import argparse
import logging
from torch import nn

sys.path.append("../../")
from fedlab.core.network import DistNetwork
from fedlab.contrib.algorithm.basic_server import AsyncServerHandler
from fedlab.core.server.manager import AsynchronousServerManager
from fedlab.models import MLP


def setup_logging(log_file="server.log"):
    """
    Sets up the logging configuration for the server.
    
    Args:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def save_global_model_snapshot(model, round_num, snapshot_dir="./global_snapshots"):
    """
    Saves a snapshot of the global model at a specific training round.
    
    Args:
        model (torch.nn.Module): The global model to save.
        round_num (int): The current round number.
        snapshot_dir (str): Directory to save the snapshots.
    """
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    snapshot_path = os.path.join(snapshot_dir, f"global_model_round_{round_num}.pth")
    torch.save(model.state_dict(), snapshot_path)
    logging.info(f"Global model snapshot saved at round {round_num} to {snapshot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asynchronous Federated Learning Server")

    # Command-line arguments for server configuration
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address of the server")
    parser.add_argument("--port", type=str, default="3002", help="Port number for communication")
    parser.add_argument("--world_size", type=int, required=True, help="Number of devices in the federation")
    parser.add_argument("--global_rounds", type=int, default=10, help="Number of global training rounds")
    parser.add_argument("--learning_rate", type=float, default=0.5, help="Learning rate for global model updates")
    parser.add_argument("--log_file", type=str, default="server.log", help="Log file path")

    args = parser.parse_args()

    # Set up logging
    setup_logging(log_file=args.log_file)
    logging.info("Starting the Federated Learning Server...")

    # Initialize the global model
    model = MLP(784, 10)
    logging.info("Initialized global model (MLP with input=784, output=10).")

    # Initialize the server handler
    handler = AsyncServerHandler(model, global_round=args.global_rounds)
    handler.setup_optim(args.learning_rate)
    logging.info(f"Server handler configured with {args.global_rounds} global rounds and "
                 f"learning rate {args.learning_rate}.")

    # Set up the network for communication
    network = DistNetwork(address=(args.ip, args.port), world_size=args.world_size, rank=0)
    logging.info(f"Network configured with IP={args.ip}, Port={args.port}, World Size={args.world_size}.")

    # Initialize the server manager
    Manager = AsynchronousServerManager(handler=handler, network=network)
    logging.info("Asynchronous Server Manager initialized.")

    # Run the server manager
    try:
        for round_num in range(args.global_rounds):
            Manager.run()
            logging.info(f"Completed global round {round_num + 1}/{args.global_rounds}.")
            save_global_model_snapshot(model, round_num + 1)  # Save model snapshot after each round
    except KeyboardInterrupt:
        logging.info("Federated Learning Server interrupted by user.")
    except Exception as e:
        logging.error(f"An error occurred during server execution: {e}")
    finally:
        logging.info("Federated Learning Server shutting down.")

