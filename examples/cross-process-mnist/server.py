import argparse
import os
import logging
import sys
from torch import nn
from datetime import datetime

sys.path.append("../../")
from fedlab.utils.logger import Logger
from fedlab.models.mlp import MLP
from fedlab.contrib.algorithm.fedavg import FedAvgServerHandler
from fedlab.core.server.manager import SynchronousServerManager
from fedlab.core.network import DistNetwork

def setup_logging(log_file="server.log"):
    """
    Sets up logging configuration for the server.

    Args:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def save_global_model_snapshot(model, round_num, snapshot_dir="./snapshots"):
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
    logging.info(f"Global model snapshot saved for round {round_num} at {snapshot_path}")

def monitor_server_resources():
    """
    Logs server resource usage statistics such as CPU and memory usage.
    """
    try:
        import psutil
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        logging.info(f"Server CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")
    except ImportError:
        logging.warning("psutil module not installed. Skipping resource monitoring.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL server example with enhancements")

    # Network Configuration
    parser.add_argument('--ip', type=str, help="IP address of the server")
    parser.add_argument('--port', type=str, help="Port for server communication")
    parser.add_argument('--world_size', type=int, help="Number of participating devices")
    parser.add_argument('--ethernet', type=str, default=None, help="Ethernet interface (optional)")

    # FL Configuration
    parser.add_argument('--round', type=int, default=3, help="Number of global training rounds")
    parser.add_argument('--sample', type=float, default=1, help="Client sampling ratio per round")

    # Enhanced Features
    parser.add_argument('--log_file', type=str, default="server.log", help="Path to the log file")
    parser.add_argument('--snapshot_dir', type=str, default="./snapshots", help="Directory to save model snapshots")
    args = parser.parse_args()

    # Setup Logging
    setup_logging(log_file=args.log_file)
    logging.info("Initializing Federated Learning Server")

    # Define Model and Handler
    model = MLP(784, 10)
    LOGGER = Logger(log_name="server")
    handler = FedAvgServerHandler(
        model,
        global_round=args.round,
        logger=LOGGER,
        sample_ratio=args.sample
    )

    # Setup Network Communication
    network = DistNetwork(
        address=(args.ip, args.port),
        world_size=args.world_size,
        rank=0,
        ethernet=args.ethernet
    )

    # Setup Server Manager
    manager_ = SynchronousServerManager(
        handler=handler,
        network=network,
        mode="GLOBAL",
        logger=LOGGER
    )

    # Run Federated Learning
    try:
        for current_round in range(1, args.round + 1):
            logging.info(f"Starting round {current_round}/{args.round}")
            manager_.run()  # Start the federated learning round

            # Monitor resources after each round
            monitor_server_resources()

            # Save global model checkpoint
            save_global_model_snapshot(model, current_round, snapshot_dir=args.snapshot_dir)
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")

