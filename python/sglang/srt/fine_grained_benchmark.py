import json
import os
import shutil
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _compute_dir_output():
    dir_raw = os.environ.get("SGLANG_FINE_GRAINED_BENCHMARK_DIR")
    if not dir_raw:
        return None
    return str(Path(dir_raw) / "fine_grained_benchmark")


_dir_output = _compute_dir_output()


def is_enabled():
    return _dir_output is not None


def maybe_benchmark(forward_batch: "ForwardBatch", tp_rank: int):
    return benchmark(forward_batch, tp_rank) if is_enabled() else nullcontext()


@contextmanager
def benchmark(forward_batch: "ForwardBatch", tp_rank: int):
    import nvtx

    num_tokens = forward_batch.input_ids.shape[0]
    global_num_tokens = (
        sum(forward_batch.global_num_tokens_cpu)
        if forward_batch.global_num_tokens_cpu is not None
        else None
    )
    debug_name = (
        f"forward"
        f"-{forward_batch.forward_mode.name}"
        f"-bs{forward_batch.batch_size}"
        f"-tok{num_tokens}"
        f"-gtok{global_num_tokens}"
    )

    torch.cuda.synchronize()
    start_time = time.time()
    with nvtx.annotate(debug_name):
        try:
            yield
        finally:
            torch.cuda.synchronize()
            latency = time.time() - start_time
            debug_info = dict(
                forward_mode=forward_batch.forward_mode.name,
                throughput=num_tokens / latency,
                latency=latency,
                batch_size=forward_batch.batch_size,
                num_tokens=num_tokens,
                global_num_tokens=global_num_tokens,
                # can_run_tbo=forward_batch.can_run_tbo, # TODO enable in tbo PR
                start_time=start_time,
                tp_rank=tp_rank,
            )
            _write_output(debug_info, tp_rank)


def _write_output(data, tp_rank):
    path = Path(_dir_output) / f"TP{tp_rank}.jsonl"
    with path.open("a") as fp:
        fp.write(f"{json.dumps(data)}\n")


def clear_output():
    shutil.rmtree(_dir_output, ignore_errors=True)
    Path(_dir_output).mkdir(parents=True, exist_ok=True)


def read_output() -> List[Dict[str, Any]]:
    return [
        json.loads(row)
        for path in sorted(list(Path(_dir_output).glob("*.jsonl")))
        for row in path.read_text().split("\n")
        if row
    ]
