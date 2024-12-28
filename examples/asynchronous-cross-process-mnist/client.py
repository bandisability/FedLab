import torchvision
import torchvision.transforms as transforms
import torch
import argparse
import sys
import os
import logging
import psutil

from torch import nn

sys.path.append("../../")
from fedlab.core.client.manager import ActiveClientManager
from fedlab.core.network import DistNetwork
from fedlab.contrib.algorithm.basic_client import SGDClientTrainer
from fedlab.models import MLP
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST

# ===================== Enhanced Trainer Class =====================
class EnhancedAsyncTrainer(SGDClientTrainer):
    """
    Enhanced AsyncTrainer with resource-aware adjustments and detailed logging.
    """
    def __init__(self, model, cuda=False):
        super(EnhancedAsyncTrainer, self).__init__(model, cuda)
        self.round = 0

    @property
    def uplink_package(self):
        return [self.model_parameters, self.round]

    def local_process(self, payload, id):
        """
        Process incoming model parameters and train locally.
        """
        model_parameters = payload[0]
        self.round = payload[1]
        train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.train_with_dynamic_adjustments(model_parameters, train_loader)

    def train_with_dynamic_adjustments(self, model_parameters, train_loader):
        """
        Train with dynamic batch size and resource-aware adjustments.
        """
        resources = self.monitor_system_resources()
        adjusted_batch_size = self.adjust_batch_size(resources, self.batch_size)
        adjusted_epochs = self.adjust_local_epochs(resources, self.epochs)

        logging.info(f"Adjusted batch size: {adjusted_batch_size}, Adjusted epochs: {adjusted_epochs}")
        self.batch_size = adjusted_batch_size
        self.epochs = adjusted_epochs

        self.train(model_parameters, train_loader)

    def adjust_batch_size(self, resources, base_batch_size):
        """
        Dynamically adjusts batch size based on CPU and memory utilization.
        """
        cpu_usage = resources.get("cpu", 50)
        memory_usage = resources.get("memory", 50)

        if cpu_usage > 80 or memory_usage > 80:
            return max(1, base_batch_size // 2)
        elif cpu_usage < 50 and memory_usage < 50:
            return base_batch_size * 2
        return base_batch_size

    def adjust_local_epochs(self, resources, base_epochs):
        """
        Dynamically adjusts local training epochs based on resource availability.
        """
        cpu_usage = resources.get("cpu", 50)
        memory_usage = resources.get("memory", 50)

        if cpu_usage > 80 or memory_usage > 80:
            return max(1, base_epochs // 2)
        elif cpu_usage < 50 and memory_usage < 50:
            return base_epochs * 2
        return base_epochs

    def monitor_system_resources(self):
        """
        Monitors system resources including CPU and memory usage.
        """
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        return {"cpu": cpu_usage, "memory": memory_usage}

    def save_model_snapshot(self, step, snapshot_dir="./snapshots"):
        """
        Saves a snapshot of the model at a specific training step.
        """
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        snapshot_path = os.path.join(snapshot_dir, f"model_snapshot_round_{step}.pth")
        torch.save(self.model.state_dict(), snapshot_path)
        logging.info(f"Model snapshot saved at round {step} to {snapshot_path}")

# ===================== Argument Parser =====================
parser = argparse.ArgumentParser(description='Distributed Federated Learning Example')
parser.add_argument('--ip', type=str, default='127.0.0.1', help="Server IP address")
parser.add_argument('--port', type=str, default='3002', help="Server port")
parser.add_argument('--world_size', type=int, help="Total number of clients")
parser.add_argument('--rank', type=int, help="Rank of the current client")
parser.add_argument('--epochs', type=int, default=2, help="Number of local training epochs")
parser.add_argument('--lr', type=float, default=0.1, help="Learning rate")
parser.add_argument('--batch_size', type=int, default=100, help="Batch size for local training")
parser.add_argument('--log_file', type=str, default="training.log", help="Path to the log file")
args = parser.parse_args()

# ===================== Logging Setup =====================
logging.basicConfig(
    filename=args.log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting Federated Learning...")

# ===================== Main Execution =====================
if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

# Define model and dataset
model = MLP(784, 10)
trainer = EnhancedAsyncTrainer(model, cuda=args.cuda)
dataset = PathologicalMNIST(root='../../datasets/mnist/', path="../../datasets/mnist/")

# Preprocess data if this is the server node
if args.rank == 1:
    dataset.preprocess()

trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# Define network communication
network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=args.rank)

# Start federated learning
Manager = ActiveClientManager(trainer=trainer, network=network)
Manager.run()
