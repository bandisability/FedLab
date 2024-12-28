import argparse
import os
import sys
from statistics import mode

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from fedlab.core.client.manager import PassiveClientManager
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger
from fedlab.models.mlp import MLP
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.contrib.algorithm.fedavg import FedAvgClientTrainer

# Enhancements: Add logging setup, model checkpointing, and resource monitoring
import logging
from datetime import datetime

def setup_logging(log_file="client.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def save_model_checkpoint(model, epoch, checkpoint_dir="./checkpoints"):
    """Save model checkpoint."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Checkpoint saved at {checkpoint_path}")

def monitor_resources():
    """Monitor system resources such as CPU and memory usage."""
    import psutil
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    logging.info(f"Resource Monitoring - CPU: {cpu_usage}%, Memory: {memory_usage}%")

# Initialize argparse for command-line arguments
parser = argparse.ArgumentParser(description="Enhanced Federated Learning Client")

# Networking arguments
parser.add_argument("--ip", type=str, help="Server IP address")
parser.add_argument("--port", type=str, help="Server port")
parser.add_argument("--world_size", type=int, help="Total number of devices in the federation")
parser.add_argument("--rank", type=int, help="Rank of this client")
parser.add_argument("--ethernet", type=str, default=None, help="Ethernet device to use for communication")

# Training arguments
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--epochs", type=int, default=2, help="Number of local training epochs")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
args = parser.parse_args()

# Enable CUDA if available
args.cuda = torch.cuda.is_available()

# Initialize the model
model = MLP(784, 10)

# Initialize the network for communication
network = DistNetwork(
    address=(args.ip, args.port),
    world_size=args.world_size,
    rank=args.rank,
    ethernet=args.ethernet,
)

# Setup logging
log_file = f"client_{args.rank}.log"
setup_logging(log_file)

logging.info("Initializing Federated Learning Client")

# Log device info
if args.cuda:
    logging.info(f"CUDA available. Using GPU.")
else:
    logging.info(f"CUDA not available. Using CPU.")

# Trainer setup
trainer = FedAvgClientTrainer(model, cuda=args.cuda)

# Dataset setup
dataset = PathologicalMNIST(
    root="../../datasets/mnist/",
    path="../../datasets/mnist/",
    num_clients=args.world_size - 1,
)
if args.rank == 1:
    dataset.preprocess()
    logging.info("Dataset preprocessing completed.")

trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# Initialize manager
LOGGER = Logger(log_name="client " + str(args.rank))
manager_ = PassiveClientManager(trainer=trainer, network=network, logger=LOGGER)

# Run federated learning
try:
    for epoch in range(args.epochs):
        monitor_resources()  # Monitor system resources
        manager_.run()
        save_model_checkpoint(model, epoch, checkpoint_dir=args.checkpoint_dir)  # Save checkpoint after each epoch
        logging.info(f"Epoch {epoch + 1}/{args.epochs} completed successfully.")
except Exception as e:
    logging.error(f"An error occurred: {str(e)}")

