"""Keep one lazy CUDA GradScaler per optimizer, including across checkpoints."""

import torch


def get_grad_scaler(optimizer):
    scaler = getattr(optimizer, "_amp_grad_scaler", None)
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda")
        saved = getattr(optimizer, "_amp_grad_scaler_state", None)
        if saved:
            scaler.load_state_dict(saved)
        optimizer._amp_grad_scaler = scaler
    return scaler


def grad_scaler_state_dict(optimizer):
    scaler = getattr(optimizer, "_amp_grad_scaler", None)
    if scaler is not None:
        return scaler.state_dict()
    return getattr(optimizer, "_amp_grad_scaler_state", None)


def load_grad_scaler_state_dict(optimizer, state):
    optimizer._amp_grad_scaler_state = state
    if hasattr(optimizer, "_amp_grad_scaler"):
        del optimizer._amp_grad_scaler
