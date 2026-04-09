import logging

import torch
import comfy.model_management

log = logging.getLogger("ComfyUI-OmniXPU")

_omni_norm = None
_logged_first_use = False


def _can_use_omni(x):
    if _omni_norm is None or not x.is_xpu:
        return False
    if x.ndim < 2:
        return False
    h = x.shape[-1]
    return h <= 8192 and h % 32 == 0


def _log_first(op, shape):
    global _logged_first_use
    if not _logged_first_use:
        _logged_first_use = True
        log.info("[OmniXPU] norm first use: %s shape=%s", op, shape)


def apply():
    global _omni_norm
    import sys
    probe = sys.modules.get("ComfyUI-OmniXPU.probe")
    if probe.norm is None:
        return False, "omni_xpu_kernel norm not available"
    _omni_norm = probe.norm

    import comfy.ops as comfy_ops

    # --- LayerNorm ---
    LN = comfy_ops.disable_weight_init.LayerNorm
    _orig_ln_cast = LN.forward_comfy_cast_weights
    _orig_ln_fwd = LN.forward

    def _ln_cast(self, input):
        if _can_use_omni(input) and len(self.normalized_shape) == 1 and self.weight is not None:
            _log_first("LayerNorm", input.shape)
            weight, bias = comfy_ops.cast_bias_weight(self, input)[:2]
            orig = input.shape
            x = _omni_norm.layer_norm(input.reshape(-1, orig[-1]), weight, bias, self.eps)
            return x.reshape(orig)
        return _orig_ln_cast(self, input)

    def _ln_fwd(self, *args, **kwargs):
        if self.comfy_cast_weights or len(self.weight_function) > 0:
            return _ln_cast(self, *args, **kwargs)
        input = args[0] if args else kwargs.get("input")
        if input is not None and _can_use_omni(input) and len(self.normalized_shape) == 1 and self.weight is not None:
            _log_first("LayerNorm", input.shape)
            orig = input.shape
            w = self.weight
            b = self.bias
            x = _omni_norm.layer_norm(input.reshape(-1, orig[-1]), w, b, self.eps)
            return x.reshape(orig)
        return _orig_ln_fwd(self, *args, **kwargs)

    LN.forward_comfy_cast_weights = _ln_cast
    LN.forward = _ln_fwd

    # --- RMSNorm ---
    RN = comfy_ops.disable_weight_init.RMSNorm
    _orig_rn_cast = RN.forward_comfy_cast_weights
    _orig_rn_fwd = RN.forward

    def _rn_cast(self, input):
        if _can_use_omni(input) and self.weight is not None:
            _log_first("RMSNorm", input.shape)
            weight = comfy_ops.cast_bias_weight(self, input)[0]
            eps = self.eps if self.eps is not None else 1e-6
            orig = input.shape
            x = _omni_norm.rms_norm(weight, input.reshape(-1, orig[-1]), eps)
            return x.reshape(orig)
        return _orig_rn_cast(self, input)

    def _rn_fwd(self, *args, **kwargs):
        if self.comfy_cast_weights or len(self.weight_function) > 0:
            return _rn_cast(self, *args, **kwargs)
        input = args[0] if args else kwargs.get("input")
        if input is not None and _can_use_omni(input) and self.weight is not None:
            _log_first("RMSNorm", input.shape)
            eps = self.eps if self.eps is not None else 1e-6
            orig = input.shape
            x = _omni_norm.rms_norm(self.weight, input.reshape(-1, orig[-1]), eps)
            return x.reshape(orig)
        return _orig_rn_fwd(self, *args, **kwargs)

    RN.forward_comfy_cast_weights = _rn_cast
    RN.forward = _rn_fwd

    # --- functional rms_norm ---
    try:
        import comfy.rmsnorm as comfy_rmsnorm
        _orig_rms_fn = comfy_rmsnorm.rms_norm

        def _patched_rms_norm(x, weight=None, eps=1e-6):
            if _can_use_omni(x):
                _log_first("rms_norm_fn", x.shape)
                orig = x.shape
                x_2d = x.reshape(-1, orig[-1])
                if weight is not None:
                    w = comfy.model_management.cast_to(weight, dtype=x.dtype, device=x.device)
                else:
                    w = torch.ones(orig[-1], dtype=x.dtype, device=x.device)
                return _omni_norm.rms_norm(w, x_2d, eps).reshape(orig)
            return _orig_rms_fn(x, weight=weight, eps=eps)

        comfy_rmsnorm.rms_norm = _patched_rms_norm
    except (ImportError, AttributeError):
        pass  # comfy.rmsnorm may not exist in all versions

    return True, None
