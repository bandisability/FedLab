# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import torch
from torch.multiprocessing import Queue
import logging

from ..network_manager import NetworkManager
from ..communicator.processor import Package, PackageProcessor
from ..network import DistNetwork
from ..server.handler import ParameterServerBackendHandler

from ...utils.logger import Logger
from ...utils.message_code import MessageCode

DEFAULT_SERVER_RANK = 0


class ServerSynchronousManager(NetworkManager):
    """Synchronous communication

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronously communicate with clients following agreements defined in :meth:`run`.

    Args:
        handler (ParameterServerBackendHandler, optional): Backend calculation handler for parameter server.
        network (DistNetwork): Manage ``torch.distributed`` network communication.
        logger (Logger, optional): :attr:`logger` for server handler. If set to ``None``, none logging output files will be generated while only on screen. Default: ``None``.
    """

    def __init__(self, handler, network, logger=None):

        super(ServerSynchronousManager, self).__init__(network, handler)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

    def run(self):
        """Main Process:
            1. Network initialization.

            2. Loop:
                2.1 activate clients.

                2.2 listen for message from clients -> transmit received parameters to server backend.

            3. Stop loop when stop condition is satisfied.

            4. Shut down clients, then close network connection.

        Note:
            user can overwrite this function to customize main process of Server.
        """

        self.setup()
        while self._handler.stop_condition() is not True:

            activate = threading.Thread(target=self.activate_clients)
            activate.start()

            # waiting for packages
            while True:
                sender, message_code, payload = PackageProcessor.recv_package()
                if self.on_receive(sender, message_code, payload):
                    break

        self.shutdown_clients()
        self._network.close_network_connection()

    def on_receive(self, sender, message_code, payload):
        """Actions to perform in server when receiving a package from one client.

        Server transmits received package to backend computation handler for aggregation or others
        manipulations.

        Note:
            Communication agreements related: user can overwrite this function to customize
            communication agreements. This method is key component connecting behaviors of
            :class:`ParameterServerBackendHandler` and :class:`NetworkManager`.


        Args:
            sender (int): Rank of sender client process.
            message_code (MessageCode): Predefined communication message code.
            payload (list[torch.Tensor]): A list of tensor, unpacked package received from clients.

        Raises:
            Exception: Unexpected :class:`MessageCode`.
        """
        if message_code == MessageCode.ParameterUpdate:
            model_parameters = payload[0]
            update_flag = self._handler.add_model(sender, model_parameters)
            return update_flag
        else:
            raise Exception("Unexpected message code {}".format(message_code))

    def activate_clients(self):
        """Activate subset of clients to join in one FL round

        Manager will start a new thread to send activation package to chosen clients' process rank.
        The ranks of clients are obtained from :meth:`handler.sample_clients`.

        Note:
            Communication agreements related: User can overwrite this function to customize
            activation package.
        """
        clients_this_round = self._handler.sample_clients()
        self._LOGGER.info(
            "client id list for this FL round: {}".format(clients_this_round)
        )

        for client_idx in clients_this_round:
            model_params = self._handler.model_parameters  # serialized model params
            pack = Package(
                message_code=MessageCode.ParameterUpdate, content=model_params
            )
            PackageProcessor.send_package(pack, dst=client_idx)

    def shutdown_clients(self):
        """Shut down all clients.

        Send package to every client with :attr:`MessageCode.Exit` to ask client to exit.

        Note:
            Communication agreements related: User can overwrite this function to define package
            for exiting information.

        """
        for client_idx in range(1, self._handler.client_num_in_total + 1):
            pack = Package(message_code=MessageCode.Exit)
            PackageProcessor.send_package(pack, dst=client_idx)


class ServerAsynchronousManager(NetworkManager):
    """Asynchronous communication network manager for server

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Asynchronously communicate with clients following agreements defined in :meth:`run`.

    Args:
        handler (ParameterServerBackendHandler, optional): Backend computation handler for parameter server.
        network (DistNetwork): Manage ``torch.distributed`` network communication.
        logger (Logger, optional): :attr:`logger` for server handler. If set to ``None``, none logging output files will be generated while only on screen. Default: ``None``.
    """

    def __init__(self, handler, network, logger=None):

        super(ServerAsynchronousManager, self).__init__(network, handler)

        if logger is None:
            logging.getLogger().setLevel(logging.INFO)
            self._LOGGER = logging
        else:
            self._LOGGER = logger

        self.message_queue = Queue()

    def run(self):
        """Main process"""
        self.setup()

        watching = threading.Thread(target=self.watching_queue)
        watching.start()

        while self._handler.stop_condition() is not True:
            sender, message_code, payload = PackageProcessor.recv_package()
            self.on_receive(sender, message_code, payload)

        self.shutdown_clients()
        self._network.close_network_connection()

    def on_receive(self, sender, message_code, payload):
        """Communication agreements of asynchronous FL.

        - Server receive ParameterRequest from client. Send model parameter to client.
        - Server receive ParameterUpdate from client. Transmit parameters to queue waiting for aggregation.

        Args:
            sender (int): Rank of sender client process.
            message_code (MessageCode): message code
            payload (list[torch.Tensor]): List of tensors.

        Raises:
            ValueError: [description]
        """
        if message_code == MessageCode.ParameterRequest:
            pack = Package(message_code=MessageCode.ParameterUpdate)
            model_params = self._handler.model_parameters
            pack.append_tensor_list(
                [model_params, torch.Tensor(self._handler.server_time)]
            )
            self._LOGGER.info(
                "Send model to rank {}, current server model time is {}".format(
                    sender, self._handler.server_time
                )
            )
            PackageProcessor.send_package(pack, dst=sender)

        elif message_code == MessageCode.ParameterUpdate:
            self.message_queue.put((sender, message_code, payload))

        else:
            raise ValueError("Unexpected message code {}".format(message_code))

    def watching_queue(self):
        """Asynchronous communication maintain a message queue. A new thread will be started to run this function.

        Note:
            Customize strategy by overwriting this function.
        """
        while self._handler.stop_condition() is not True:
            _, _, payload = self.message_queue.get()
            parameters = payload[0]
            model_time = payload[1]
            self._handler._update_model(parameters, model_time)

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to clients with ``MessageCode.Exit``.

        Note:
            Communication agreements related: user can overwrite this function to define close
            package.

        """
        for client_idx in range(1, self._handler.client_num_in_total + 1):
            _, message_code, _ = PackageProcessor.recv_package(src=client_idx)
            if message_code == MessageCode.ParameterUpdate:
                PackageProcessor.recv_package(
                    src=client_idx
                )  # the next package is model request
            pack = Package(message_code=MessageCode.Exit)
            PackageProcessor.send_package(pack, dst=client_idx)