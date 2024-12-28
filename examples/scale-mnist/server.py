import sys
import argparse
import os
import logging
import torch
from torch import nn
from torchvision import transforms

# Extend system path for relative imports
sys.path.append("../../")

# Import FedLab core components
from fedlab.core.server.manager import SynchronousServerManager
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.core.network import DistNetwork
from fedlab.models.mlp import MLP
from fedlab.utils.logger import Logger
from fedlab.utils.functional import AverageMeter, evaluate

# ========================== Argument Parsing ==========================
parser = argparse.ArgumentParser(description="Optimized Federated Learning Server")

parser.add_argument("--ip", type=str, default="127.0.0.1", help="Server IP address")
parser.add_argument("--port", type=str, default="3002", help="Server port number")
parser.add_argument("--world_size", type=int, required=True, help="Total number of clients + server")
parser.add_argument("--ethernet", type=str, default=None, help="Network interface")

parser.add_argument("--round", type=int, default=10, help="Number of global training rounds")
parser.add_argument("--sample", type=float, default=1, help="Client sampling ratio per round")
parser.add_argument("--log_file", type=str, default="server.log", help="Path to the server log file")
parser.add_argument("--checkpoint_dir", type=str, default="./server_checkpoints", help="Directory for saving model checkpoints")

args = parser.parse_args()

# ========================== Logging Setup ==========================
def setup_logging(log_file):
    """Configures logging for the server."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Server logging initialized.")

setup_logging(args.log_file)

# ========================== Model and Handler Setup ==========================
logging.info("Initializing Federated Learning Server")
model = MLP(784, 10)
handler = SyncServerHandler(
    model=model,
    global_round=args.round,
    sample_ratio=args.sample,
    cuda=torch.cuda.is_available(),
)

# ========================== Network Communication Setup ==========================
network = DistNetwork(
    address=(args.ip, args.port),
    world_size=args.world_size,
    rank=0,
    ethernet=args.ethernet,
)

# ========================== Checkpointing ==========================
def save_model_checkpoint(model, current_round, checkpoint_dir):
    """Saves model checkpoint after each training round."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"global_model_round_{current_round}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Model checkpoint saved: {checkpoint_path}")

# ========================== Server Manager Setup ==========================
manager_ = SynchronousServerManager(
    network=network,
    handler=handler,
    mode="GLOBAL",
)

# ========================== Main Execution Loop ==========================
try:
    logging.info("Starting Federated Learning Process")
    for current_round in range(1, args.round + 1):
        logging.info(f"Starting Round {current_round}/{args.round}")
        manager_.run()  # Execute federated learning round

        # Save checkpoint after each round
        save_model_checkpoint(handler.model, current_round, args.checkpoint_dir)

except Exception as e:
    logging.error(f"An error occurred during training: {str(e)}")

