"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""A scheduler that manages a tensor parallel GPU worker."""

import logging
import multiprocessing

import zmq

from sglang.srt.managers.tp_worker import ModelTpServer
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import broadcast_pyobj, configure_logger, kill_parent_process
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class Scheduler:
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
    ):
        # Parse args
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size

        # Init inter-process communication
        context = zmq.Context(2)

        if self.tp_rank == 0:
            self.recv_from_tokenizer = context.socket(zmq.PULL)
            self.recv_from_tokenizer.bind(f"tcp://127.0.0.1:{port_args.scheduler_port}")

            self.send_to_detokenizer = context.socket(zmq.PUSH)
            self.send_to_detokenizer.connect(
                f"tcp://127.0.0.1:{port_args.detokenizer_port}"
            )
        else:
            self.send_to_detokenizer = None

        # Launch a tp server
        self.tp_server = ModelTpServer(
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            server_args=server_args,
            nccl_port=port_args.nccl_ports[0],
        )
        self.tp_cpu_group = self.tp_server.model_runner.tp_group.cpu_group

    def event_loop(self):
        while True:
            if self.tp_rank == 0:
                recv_reqs = self.recv_requests_from_zmq()
            else:
                recv_reqs = None

            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_rank, self.tp_cpu_group)
            out_pyobjs = self.tp_server.exposed_step(recv_reqs)

            if self.tp_rank == 0:
                for obj in out_pyobjs:
                    self.send_to_detokenizer.send_pyobj(obj)

    def recv_requests_from_zmq(self):
        recv_reqs = []

        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)

        return recv_reqs


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    pipe_writer: multiprocessing.connection.Connection,
):
    configure_logger(server_args, prefix=f" TP{tp_rank}")

    try:
        scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank)
        pipe_writer.send("ready")
        scheduler.event_loop()
    except Exception:
        msg = get_exception_traceback()
        logger.error(msg)
        kill_parent_process()
