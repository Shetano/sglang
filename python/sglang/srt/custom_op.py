from typing import Optional

import torch
from sglang.srt.utils import is_cuda, is_hip
from torch import nn

_is_cuda = is_cuda()
_is_hip = is_hip()


class CustomOp(nn.Module):
    def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def set_torch_compile_mode(self, reverse: bool):
        if reverse:
            self._forward_method = self.forward_cuda
            self.is_torch_compile = False
        else:
            # NOTE: Temporarily workaround MoE
            if "FusedMoE" in self.__class__.__name__:
                if num_tokens == 1:
                    # The performance of torch.compile on this layer is not always good when bs > 1,
                    # so we decide to only use torch.compile when bs =1
                    self._forward_method = fused_moe_forward_native
            else:
                self._forward_method = self.forward_native
            self.is_torch_compile = True

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self):
        if _is_cuda:
            return self.forward_cuda
        elif _is_hip:
            return self.forward_hip
        else:
            return self.forward_native


if _is_cuda:
    from sgl_kernel import sgl_per_tensor_quant_fp8, sgl_per_token_quant_fp8


    def scaled_fp8_quant(
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        num_token_padding: Optional[int] = None,
        use_per_token_if_dynamic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input tensor to FP8 (8-bit floating point) format.

        Args:
            input (torch.Tensor): Input tensor to be quantized
            scale (Optional[torch.Tensor]): Pre-computed scaling factor for static quantization.
                If None, scales will be computed dynamically.
            num_token_padding (Optional[int]): If specified, pad the first dimension
                of the output to at least this value.
            use_per_token_if_dynamic (bool): When using dynamic scaling (scale=None),
                determines the quantization granularity:
                - True: compute scale per token
                - False: compute single scale per tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - quantized_tensor: The FP8 quantized version of input
                - scale_tensor: The scaling factors used for quantization

        Raises:
            AssertionError: If input is not 2D or if static scale's numel != 1
        """
        assert input.ndim == 2, f"Expected 2D input tensor, got {input.ndim}D"
        shape = input.shape
        out_dtype = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
        if num_token_padding:
            shape = (max(num_token_padding, input.shape[0]), shape[1])
        output = torch.empty(shape, device=input.device, dtype=out_dtype)

        if scale is None:
            # Dynamic scaling
            if use_per_token_if_dynamic:
                scale = torch.empty(
                    (shape[0], 1), device=input.device, dtype=torch.float32
                )
                sgl_per_token_quant_fp8(input, output, scale)
            else:
                scale = torch.zeros(1, device=input.device, dtype=torch.float32)
                sgl_per_tensor_quant_fp8(
                    input, output, scale, is_static=False
                )  # False for dynamic
        else:
            # Static scaling
            assert (
                scale.numel() == 1
            ), f"Expected scalar scale, got numel={scale.numel()}"
            sgl_per_tensor_quant_fp8(
                input, output, scale, is_static=True
            )  # True for static

        return output, scale
