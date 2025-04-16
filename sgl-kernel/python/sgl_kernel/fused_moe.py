import torch
from typing import Optional
import functools
import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, moe_align_block_size, try_get_optimal_moe_config)
from vllm.scalar_type import scalar_types
from vllm.utils import direct_register_custom_op
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (marlin_quantize)
from vllm.scalar_type import scalar_types
from sglang.srt.layers.moe.topk import select_experts
from sgl_kernel import fused_marlin_moe


def fused_moe(a, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, num_bits, score, w1_zp, w2_zp, g_idx1 = None, g_idx2 = None, sort_indices1 = None, sort_indices2 = None):
    e_map = None
    marlin_output = fused_marlin_moe(
        a,
        w1,
        w2,
        w1_scale,
        w2_scale,
        score,
        topk_weights,
        topk_ids,
        global_num_experts=w1.shape[0],
        expert_map=e_map,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=w1_zp,
        w2_zeros=w2_zp,
        num_bits=num_bits,
        is_k_full=True)
    return marlin_output

def get_scalar_type(num_bits: int, has_zp: bool):
    if has_zp:
        return scalar_types.uint4 if num_bits == 4 else scalar_types.uint8
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128

def fused_marlin_moe(hidden_states: torch.Tensor,
                     w1: torch.Tensor,
                     w2: torch.Tensor,
                     w1_scale: torch.Tensor,
                     w2_scale: torch.Tensor,
                     gating_output: torch.Tensor,
                     topk_weights: torch.Tensor,
                     topk_ids: torch.Tensor,
                     global_num_experts: int = -1,
                     expert_map: Optional[torch.Tensor] = None,
                     g_idx1: Optional[torch.Tensor] = None,
                     g_idx2: Optional[torch.Tensor] = None,
                     sort_indices1: Optional[torch.Tensor] = None,
                     sort_indices2: Optional[torch.Tensor] = None,
                     w1_zeros: Optional[torch.Tensor] = None,
                     w2_zeros: Optional[torch.Tensor] = None,
                     workspace: Optional[torch.Tensor] = None,
                     num_bits: int = 8,
                     is_k_full: bool = True,
                     inplace: bool = False) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (bool): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.

    assert hidden_states.shape[
        1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2), "Hidden size mismatch w2"
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert num_bits in [4, 8]

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        topk_ids.shape[1],
        None,
        is_marlin=True,
    )
    config = get_config_func(M)

    block_size_m = config["BLOCK_SIZE_M"]

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = \
        moe_align_block_size(topk_ids, block_size_m, global_num_experts,
                             expert_map)

    if workspace is None:
        max_workspace_size = (max(2 * N, K) // 64) * \
            (sorted_token_ids.size(0) // block_size_m)
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_workspace_size = min(max_workspace_size, sms)
        workspace = torch.zeros(max_workspace_size,
                                dtype=torch.int,
                                device=device,
                                requires_grad=False)

    scalar_type1 = get_scalar_type(num_bits, w1_zeros is not None)
    scalar_type2 = get_scalar_type(num_bits, w2_zeros is not None)

    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache13 = torch.empty(
        (M * topk_ids.shape[1] * max(2 * N, K), ),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[:M * topk_ids.shape[1] * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N)
    intermediate_cache3 = intermediate_cache13[:M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)
    intermediate_cache1 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm(
        hidden_states,
        intermediate_cache1,
        w1,
        w1_scale,
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        block_size_m,         # moe_block_size
        topk,                 # top_k
        False,                # mul_topk_weights
        expert_map is not None,  # is_ep
        scalar_type1.id,      # b_q_type_id
        M,                    # size_m
        2 * N,                # size_n
        K,                    # size_k
        is_k_full,            # is_k_full
        True,                 # use_atomic_add
        True,                 # use_fp32_reduce
        False                 # is_zp_float
    )

    torch.ops._C.silu_and_mul(intermediate_cache2,
                              intermediate_cache1.view(-1, 2 * N))

    intermediate_cache3 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm(
        intermediate_cache2,
        intermediate_cache3,
        w2,
        w2_scale,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=expert_map is not None,
        b_q_type_id=scalar_type2.id,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_full_k=is_k_full,
        use_atomic_add=True,
        use_fp32_reduce=True,
        is_zp_float=False).view(-1, topk, K)

    output = hidden_states if inplace else torch.empty_like(hidden_states)
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                     dim=1,
                     out=output)