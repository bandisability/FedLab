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

from client.trainer import SerialClientTrainer
from server.handler import ServerHandler

class StandalonePipeline(object):
    def __init__(self, handler, trainer):
        """Perform standalone simulation process

        Args:
            handler (ServerHandler): _description_
            trainer (SerialClientTrainer): _description_
        """
        self.handler = handler
        self.trainer = trainer

    def main(self):
        # server side
        sampled_clients = self.handler.sample_clients()
        broadcast = self.handler.downlink_pakage
        
        # client side
        self.trainer.local_process(sampled_clients, broadcast)
        uploads = self.trainer.uplink_pakage

        # server side
        for ele in uploads:
            self.handler.load(ele)

        # evaluate
