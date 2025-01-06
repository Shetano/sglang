from types import SimpleNamespace
from typing import List

import zmq
from sglang.communicator import TypeBasedDispatcher, create_receiver, create_sender
from sglang.srt.managers.io_struct import (
    AbortReq,
    CloseSessionReqInput,
    FlushCacheReq,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    OpenSessionReqInput,
    ProfileReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import (
    Req,
)
from sglang.srt.managers.scheduler.core import SchedulerCore, SchedulerCoreCallback
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    broadcast_pyobj,
)


# TODO merge back?
class SchedulerCommunicator:
    def __init__(
        self,
        core: SchedulerCore,
        server_args: ServerArgs,
        port_args: PortArgs,
        tp_rank: int,
    ):
        self.core = core
        self.server_args = server_args
        self.tp_rank = tp_rank

        if self.tp_rank == 0 or self.server_args.enable_dp_attention:
            self._recv_from_tokenizer = create_receiver(port_args.scheduler_input_ipc_name)
            self._send_to_tokenizer = create_sender(port_args.tokenizer_ipc_name)

            if self.server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                self._send_to_detokenizer = create_sender(port_args.tokenizer_ipc_name)
            else:
                # Send to the DetokenizerManager
                self._send_to_detokenizer = create_sender(port_args.detokenizer_ipc_name)
        else:
            self._recv_from_tokenizer = None
            self._send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self._send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)

        core.callback = SchedulerCoreCallback(
            on_output=self._send_to_detokenizer.send_pyobj,
            on_event_loop_iteration=lambda: self._process_input_requests(self._recv_requests()),
        )

        self._dispatcher = TypeBasedDispatcher([
            (TokenizedGenerateReqInput, self.core.handle_generate_request),
            (TokenizedEmbeddingReqInput, self.core.handle_embedding_request),
            (FlushCacheReq, self.core.flush_cache_wrapped),
            (AbortReq, self.core.abort_request),
            (UpdateWeightFromDiskReqInput, self.core.update_weights_from_disk),
            (InitWeightsUpdateGroupReqInput, self.core.init_weights_update_group),
            (UpdateWeightsFromDistributedReqInput, self.core.update_weights_from_distributed),
            (UpdateWeightsFromTensorReqInput, self.core.update_weights_from_tensor),
            (GetWeightsByNameReqInput, self.core.get_weights_by_name),
            (ProfileReq, self.core.profile),
            (OpenSessionReqInput, self.core.open_session),
            (CloseSessionReqInput, self.core.close_session),
        ])

    def _recv_requests(self) -> List[Req]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""
        if self.tp_rank == 0 or self.server_args.enable_dp_attention:
            recv_reqs = []

            while True:
                try:
                    recv_req = self._recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
        else:
            recv_reqs = None

        if self.server_args.tp_size != 1 and not self.server_args.enable_dp_attention:
            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_rank, self.core.tp_cpu_group)
        return recv_reqs

    def _process_input_requests(self, recv_reqs: List):
        for recv_req in recv_reqs:
            output = self._dispatcher(recv_req)
            if output is not None:
                self._send_to_tokenizer.send_pyobj(output)
