import logging

import torch
import comfy.model_management

log = logging.getLogger("ComfyUI-OmniXPU")

_omni_fp8_linear = None
_logged_first = False


def apply():
    global _omni_fp8_linear
    import sys
    probe = sys.modules.get("ComfyUI-OmniXPU.probe")
    if probe.linear_fp8 is None:
        return False, "omni_xpu_kernel linear_fp8 not available"
    _omni_fp8_linear = probe.linear_fp8

    import comfy.ops as comfy_ops

    # --- Patch fp8_linear module-level function ---
    if hasattr(comfy_ops, "fp8_linear"):
        _orig_fp8_linear = comfy_ops.fp8_linear

        def _patched_fp8_linear(self, input):
            global _logged_first
            dtype = self.weight.dtype
            if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                return None

            input_shape = input.shape
            tensor_3d = input.ndim == 3
            if tensor_3d:
                input = input.reshape(-1, input_shape[2])
            if input.ndim != 2:
                return None

            if input.is_xpu:
                try:
                    lora_compute_dtype = comfy.model_management.lora_compute_dtype(input.device)
                    w, bias, offload_stream = comfy_ops.cast_bias_weight(
                        self, input, dtype=dtype, bias_dtype=input.dtype,
                        offloadable=True, compute_dtype=lora_compute_dtype, want_requant=True
                    )
                    scale_weight = getattr(self, "scale_weight", None)
                    if scale_weight is None:
                        scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
                    if not _logged_first:
                        _logged_first = True
                        log.info("[OmniXPU] fp8_gemm first use: input=%s weight=%s dtype=%s",
                                 list(input.shape), list(w.shape), dtype)
                    o = _omni_fp8_linear(input, w, scale_weight, bias)
                    comfy_ops.uncast_bias_weight(self, w, bias, offload_stream)
                    if tensor_3d:
                        o = o.reshape(input_shape[0], input_shape[1], w.shape[0])
                    return o
                except Exception as e:
                    log.info("[OmniXPU] fp8_gemm failed, falling back: %s", e)

            return _orig_fp8_linear(self, input)

        comfy_ops.fp8_linear = _patched_fp8_linear

    # --- Patch mixed_precision_ops if present ---
    # This patches the forward() method of classes returned by mixed_precision_ops()
    # to intercept FP8 weights on XPU before comfy_kitchen dispatch
    if hasattr(comfy_ops, "mixed_precision_ops"):
        _orig_mixed = comfy_ops.mixed_precision_ops

        def _patched_mixed(*args, **kwargs):
            klass = _orig_mixed(*args, **kwargs)

            _orig_fwd = klass.Linear.forward

            def _mp_forward(self, input, *fwd_args, **fwd_kwargs):
                global _logged_first
                if (_omni_fp8_linear is not None and input.is_xpu and
                        getattr(self, "quant_format", None) in ("float8_e4m3fn", "float8_e5m2") and
                        len(self.weight_function) == 0 and len(self.bias_function) == 0):
                    input_shape = input.shape
                    input_2d = input.reshape(-1, input_shape[-1]) if input.ndim == 3 else input
                    if input_2d.ndim == 2:
                        try:
                            w = self.weight
                            fp8_dtype = torch.float8_e4m3fn if self.quant_format == "float8_e4m3fn" else torch.float8_e5m2
                            # Handle QuantizedTensor
                            QuantizedTensor = getattr(comfy_ops, "QuantizedTensor", None)
                            if QuantizedTensor and isinstance(w, QuantizedTensor):
                                w_fp8 = w._qdata
                                scale_w = getattr(w.params, "scale", None)
                            else:
                                w_fp8 = w if w.dtype == fp8_dtype else w.view(fp8_dtype)
                                scale_w = getattr(self, "scale_weight", None)
                            if scale_w is None:
                                scale_w = torch.ones((), device=input.device, dtype=torch.float32)
                            scale_w = comfy.model_management.cast_to_device(scale_w, input.device, torch.float32)
                            w_fp8 = comfy.model_management.cast_to_device(w_fp8, input.device, None)
                            bias = comfy.model_management.cast_to_device(self.bias, input.device, input.dtype) if self.bias is not None else None

                            if not _logged_first:
                                _logged_first = True
                                log.info("[OmniXPU] fp8_gemm (mixed_precision) first use: input=%s weight=%s",
                                         list(input_2d.shape), list(w_fp8.shape))
                            o = _omni_fp8_linear(input_2d, w_fp8, scale_w, bias)
                            if input.ndim == 3:
                                o = o.reshape(input_shape[0], input_shape[1], -1)
                            return o
                        except Exception:
                            pass
                return _orig_fwd(self, input, *fwd_args, **fwd_kwargs)

            klass.Linear.forward = _mp_forward
            return klass

        comfy_ops.mixed_precision_ops = _patched_mixed

    return True, None
