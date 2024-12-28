import torch
import argparse
import sys
import os
import logging
from torch import nn
import torchvision
import torchvision.transforms as transforms

sys.path.append("../../")

from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.client import PassiveClientManager
from fedlab.core.network import DistNetwork
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.models.mlp import MLP
from fedlab.utils.logger import Logger

# ================================
# Argument Parsing and Setup
# ================================
parser = argparse.ArgumentParser(description="Enhanced Federated Learning Client Example")

parser.add_argument("--ip", type=str, default="127.0.0.1", help="Server IP address")
parser.add_argument("--port", type=str, default="3002", help="Server port")
parser.add_argument("--world_size", type=int, help="Total number of clients + server")
parser.add_argument("--rank", type=int, help="Rank of this client")
parser.add_argument("--ethernet", type=str, default=None, help="Ethernet device name (optional)")

parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for local training")
parser.add_argument("--epochs", type=int, default=2, help="Number of local training epochs")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size for local training")
parser.add_argument("--log_file", type=str, default="client_training.log", help="Path to the log file")
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")

args = parser.parse_args()

# ================================
# Device and Logger Setup
# ================================
if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

# Setup logging
os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
logging.basicConfig(
    filename=args.log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Initializing Federated Learning Client")

# ================================
# Model Definition
# ================================
model = MLP(784, 10)

# ================================
# Trainer Definition
# ================================
trainer = SGDSerialClientTrainer(model, num_clients=10, cuda=args.cuda)

# ================================
# Dataset Preparation
# ================================
dataset = PathologicalMNIST(
    root="../../datasets/mnist/",
    path="../../datasets/mnist/",
    num_clients=100,
)

if args.rank == 1:
    logging.info("Preprocessing dataset on rank 1")
    dataset.preprocess()

trainer.setup_dataset(dataset)
trainer.setup_optim(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

# ================================
# Network Communication Setup
# ================================
network = DistNetwork(
    address=(args.ip, args.port),
    world_size=args.world_size,
    rank=args.rank,
    ethernet=args.ethernet,
)

# ================================
# Manager Definition
# ================================
manager_ = PassiveClientManager(trainer=trainer, network=network)

# ================================
# Model Checkpointing
# ================================
def save_model_checkpoint(model, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"client_checkpoint_epoch_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Model checkpoint saved at {checkpoint_path}")


# ================================
# Main Execution Loop
# ================================
try:
    logging.info("Starting Federated Learning")
    for epoch in range(args.epochs):
        manager_.run()  # Run the federated learning round

        # Save checkpoint after each epoch
        save_model_checkpoint(model, epoch, args.checkpoint_dir)
        logging.info(f"Epoch {epoch + 1}/{args.epochs} completed successfully.")
except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
