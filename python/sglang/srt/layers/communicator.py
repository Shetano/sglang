from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Tuple

import torch
from sglang.srt.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce
from sglang.srt.layers.dp_attention import attn_tp_all_gather, dp_gather_partial, dp_scatter, attn_tp_reduce_scatter, \
    get_local_attention_dp_size, get_attention_tp_size, get_attention_tp_rank
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class ScatterMode(Enum):
    SCATTERED = auto()
    TP_ATTN_FULL = auto()
    FULL = auto()


@dataclass
class _LayerModeComputationContext:
    num_layers: int
    layer_id: int
    is_layer_sparse: bool
    is_previous_layer_sparse: Optional[bool]

    def previous_layer(self):
        assert self.is_previous_layer_sparse is not None
        return _LayerModeComputationContext(
            layer_id=self.layer_id,
            is_layer_sparse=self.is_previous_layer_sparse,
            is_previous_layer_sparse=None,

            # unchanged
            num_layers=self.num_layers,
        )


@dataclass
class LayerScatterModes:
    layer_input_mode: ScatterMode
    attn_mode: ScatterMode
    # Can be further split into e.g. ffn_input_mode and ffn_output_mode if needed
    mlp_mode: ScatterMode
    middle_residual_mode: ScatterMode
    layer_output_mode: ScatterMode

    @classmethod
    def init_new(cls, **kwargs):
        context = _LayerModeComputationContext(**kwargs)
        return cls(
            layer_input_mode=cls._compute_layer_input_mode(context),
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=cls._compute_mlp_mode(context),
            middle_residual_mode=cls._compute_middle_residual_mode(context),
            layer_output_mode=cls._compute_layer_output_mode(context),
        )

    @classmethod
    def _compute_layer_input_mode(cls, context: _LayerModeComputationContext):
        if context.layer_id == 0:
            return ScatterMode.TP_ATTN_FULL
        return cls._compute_layer_output_mode(context.previous_layer())

    @classmethod
    def _compute_mlp_mode(cls, context: _LayerModeComputationContext):
        if context.is_layer_sparse:
            return (
                ScatterMode.SCATTERED
                if global_server_args_dict["enable_deepep_moe"]
                else ScatterMode.FULL
            )
        else:
            return (
                ScatterMode.SCATTERED
                if enable_moe_dense_fully_dp()
                else ScatterMode.FULL
            )

    @classmethod
    def _compute_middle_residual_mode(cls, context: _LayerModeComputationContext):
        mlp_mode = cls._compute_mlp_mode(context)
        if mlp_mode == ScatterMode.SCATTERED:
            return ScatterMode.SCATTERED
        if mlp_mode == ScatterMode.FULL:
            return ScatterMode.TP_ATTN_FULL
        raise NotImplementedError

    @classmethod
    def _compute_layer_output_mode(cls, context: _LayerModeComputationContext):
        if context.layer_id == context.num_layers - 1:
            return ScatterMode.TP_ATTN_FULL
        return cls._compute_mlp_mode(context)


def enable_moe_dense_fully_dp():
    return global_server_args_dict["moe_dense_tp_size"] == 1


class LayerCommunicator:
    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        post_attention_layernorm: torch.nn.Module,
    ):
        self.layer_scatter_modes = layer_scatter_modes
        self.post_attention_layernorm = post_attention_layernorm

        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.tp_size = get_tensor_model_parallel_world_size()

    def forward_pre_attn(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return _communicate_simple(
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            input_mode=self.layer_scatter_modes.layer_input_mode,
            output_mode=self.layer_scatter_modes.attn_mode,
            context=TODO,
        )

    def forward_pre_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return _communicate_with_all_reduce_and_layer_norm(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            hidden_states_input_mode=self.layer_scatter_modes.attn_mode,
            residual_input_mode=self.layer_scatter_modes.layer_input_mode,
            hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,
            residual_output_mode=self.layer_scatter_modes.middle_residual_mode,
            layernorm=self.post_attention_layernorm,
            context=TODO,
        )

    def forward_layer_end(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return _communicate_summable_tensor_pair(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,
            residual_input_mode=self.layer_scatter_modes.middle_residual_mode,
            output_mode=self.layer_scatter_modes.layer_output_mode,
            context=TODO,
        )


# TODO rename?
@dataclass
class _Context:
    process_group_sizes: Dict["ScatterMode", int]
    num_tokens_of_mode: Dict["ScatterMode", int]
    attn_tp_rank: int
    attn_tp_size: int
    tp_size: int

    def is_same_group_size(self, a: "ScatterMode", b: "ScatterMode"):
        return self.process_group_sizes[a] == self.process_group_sizes[b]

    def check_shape(self, x: torch.Tensor, mode: ScatterMode):
        actual_num_tokens = x.shape[0]
        expect_num_tokens = self.num_tokens_of_mode[mode]
        assert actual_num_tokens == expect_num_tokens, f"{actual_num_tokens=} {expect_num_tokens=} {mode=} {x.shape=}"
        return x

    def check_shapes(self, xs: Tuple[torch.Tensor, ...], modes: Tuple[ScatterMode, ...]) -> Tuple[torch.Tensor, ...]:
        return tuple([self.check_shape(x, mode) for x, mode in zip(xs, modes, strict=True)])


def _communicate_simple(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    input_mode: ScatterMode,
    output_mode: ScatterMode,
    context: _Context,
) -> torch.Tensor:
    def _inner():
        nonlocal hidden_states

        if context.is_same_group_size(input_mode, output_mode):
            return hidden_states

        if (
            (input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            attn_tp_all_gather(
                list(hidden_states.tensor_split(context.attn_tp_size)), local_hidden_states
            )
            return hidden_states

        raise NotImplementedError(f"{input_mode=} {output_mode=}")

    context.check_shape(hidden_states, input_mode)
    return context.check_shape(_inner(), output_mode)


def _communicate_with_all_reduce_and_layer_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    hidden_states_input_mode: ScatterMode,
    residual_input_mode: ScatterMode,
    hidden_states_output_mode: ScatterMode,
    residual_output_mode: ScatterMode,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: _Context,
):
    """Besides communication, needs to
    1. All reduce in tp_attn_group on hidden_states
    2. Apply layer norm
    """

    def _inner():
        nonlocal hidden_states, residual

        if context.is_same_group_size(hidden_states_input_mode, hidden_states_output_mode) \
            and context.is_same_group_size(residual_input_mode, residual_output_mode):
            if context.tp_size > 1:
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
            if hidden_states.shape[0] != 0:
                hidden_states, residual = layernorm(hidden_states, residual)
            return hidden_states, residual

        if (
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (residual_input_mode == ScatterMode.TP_ATTN_FULL)
            and (hidden_states_output_mode == ScatterMode.FULL)
            and (residual_output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            if context.attn_tp_rank == 0:
                hidden_states += residual
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer,
                hidden_states,
            )
            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
            dp_scatter(residual, hidden_states, forward_batch)
            hidden_states = layernorm(hidden_states)
            return hidden_states, residual

        if (
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (residual_input_mode in [ScatterMode.SCATTERED, ScatterMode.TP_ATTN_FULL])
            and (hidden_states_output_mode == ScatterMode.SCATTERED)
            and (residual_output_mode == ScatterMode.SCATTERED)
        ):
            tensor_list = list(hidden_states.tensor_split(context.attn_tp_size))
            hidden_states = tensor_list[context.attn_tp_rank]
            attn_tp_reduce_scatter(hidden_states, tensor_list)
            if residual_input_mode == ScatterMode.TP_ATTN_FULL:
                residual = residual.tensor_split(context.attn_tp_size)[context.attn_tp_rank]
            if hidden_states.shape[0] != 0:
                hidden_states, residual = layernorm(hidden_states, residual)
            return hidden_states, residual

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {residual_output_mode=} {residual_output_mode=}")

    context.check_shapes((hidden_states, residual), (hidden_states_input_mode, residual_input_mode))
    return context.check_shapes(_inner(), (hidden_states_output_mode, residual_output_mode))


def _communicate_summable_tensor_pair(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    hidden_states_input_mode: ScatterMode,
    residual_input_mode: ScatterMode,
    output_mode: ScatterMode,
    context: _Context,
):
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""

    def _inner():
        nonlocal hidden_states, residual

        if context.is_same_group_size(hidden_states_input_mode, output_mode) \
            and context.is_same_group_size(residual_input_mode, output_mode):
            return hidden_states, residual

        if (
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.TP_ATTN_FULL)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            # TODO(ch-wan): use reduce-scatter in MLP to avoid this scatter
            # important: forward batch.gathered_buffer is used both after scatter and after gather.
            # be careful about this!
            hidden_states, global_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            dp_scatter(hidden_states, global_hidden_states, forward_batch)
            return hidden_states, residual

        if (
            (hidden_states_input_mode == ScatterMode.SCATTERED)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            hidden_states += residual
            residual = None
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            attn_tp_all_gather(
                list(hidden_states.tensor_split(context.attn_tp_size)), local_hidden_states
            )
            return hidden_states, residual

        raise NotImplementedError(f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}")

    context.check_shapes((hidden_states, residual), (hidden_states_input_mode, residual_input_mode))
    return context.check_shapes(_inner(), (output_mode, output_mode))
