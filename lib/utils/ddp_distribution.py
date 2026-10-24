"""DDP setup and communication, with single-process fallbacks."""

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def initialize_processes(backend, device):
    device = torch.device(device)
    dist.init_process_group(
        backend,
        timeout=timedelta(minutes=30),
        device_id=device if device.type == "cuda" else None,
    )


def shutdown_processes():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_process_count():
    return dist.get_world_size() if dist.is_initialized() else 1


def get_process_index():
    return dist.get_rank() if dist.is_initialized() else 0


def is_main_process():
    return get_process_index() == 0


def wrap_model_for_ddp(model, device):
    if get_process_count() == 1:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False,
    )


def wait_for_all_processes():
    if get_process_count() > 1:
        dist.barrier()


def reduce_all(values):
    """Sum tensor values across all processes in place and return the tensor."""
    if get_process_count() > 1:
        dist.all_reduce(values)
    return values


def broadcast(value):
    """Send a Python object from the main process to every process."""
    if get_process_count() > 1:
        objects = [value]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]
    return value


def gather_from_all_processes(value):
    """Collect one Python object per process, ordered by process index."""
    values = [None] * get_process_count()
    if get_process_count() > 1:
        dist.all_gather_object(values, value)
    else:
        values[0] = value
    return values


def sum_metrics_across_processes(loss, tokens, device):
    if get_process_count() > 1:
        values = torch.stack((
            torch.as_tensor(loss, dtype=torch.float64, device=device),
            torch.as_tensor(tokens, dtype=torch.float64, device=device),
        ))
        loss, tokens = reduce_all(values).tolist()
    elif isinstance(loss, torch.Tensor):
        loss = loss.item()
    return loss, int(tokens)
