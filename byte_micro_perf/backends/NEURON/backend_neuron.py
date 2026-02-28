import os
import sys
import json
import time
import pathlib
import subprocess
import importlib.metadata

import torch
import torch.distributed as dist

FILE_DIR = pathlib.Path(__file__).parent.absolute()
BACKEND_DIR = FILE_DIR.parent
MICRO_PERF_DIR = BACKEND_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.backend import Backend
from core.utils import logger

# NOTE: Do NOT import torch_xla at module level.
# Importing torch_xla triggers PJRT runtime initialization which grabs
# NeuronCores in the parent process, preventing spawned child processes
# from accessing them. All torch_xla imports must be deferred to methods
# that only execute inside child (subprocess) contexts.

try:
    from backends.NEURON.env_neuron import NEURON_ENV
except Exception:
    NEURON_ENV = {}

try:
    from backends.NEURON.provider_neuron import NEURON_PROVIDER
except Exception:
    NEURON_PROVIDER = {}


class BackendNEURON(Backend):
    def __init__(self):
        # Patch pin_memory before any tensor operations -- Neuron machines
        # have no NVIDIA driver so pin_memory() on CPU tensors fails.
        self._patch_pin_memory()

        super().__init__()

    def _patch_pin_memory(self):
        _original_pin_memory = torch.Tensor.pin_memory

        def _safe_pin_memory(tensor, device=None):
            try:
                return _original_pin_memory(tensor, device=device)
            except Exception:
                return tensor

        torch.Tensor.pin_memory = _safe_pin_memory

    # ── neuron-ls helpers ─────────────────────────────

    def _get_neuron_ls_data(self):
        try:
            result = subprocess.run(
                ["neuron-ls", "-j"],
                capture_output=True, text=True, timeout=10
            )
            return json.loads(result.stdout)
        except Exception:
            return []

    def _get_instance_type(self):
        try:
            result = subprocess.run(
                ["neuron-ls"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split("\n"):
                if "instance-type" in line:
                    return line.split(":")[1].strip()
        except Exception:
            pass
        return "unknown"

    # ── backend info ──────────────────────────────────

    def get_backend_info(self):
        info = {}

        neuron_data = self._get_neuron_ls_data()
        instance_type = self._get_instance_type()

        nc_count = 0
        total_memory = 0
        if neuron_data:
            for dev in neuron_data:
                nc_count += dev.get("nc_count", 0)
                total_memory += dev.get("memory_size", 0)

        info["device_name"] = instance_type
        info["device_count"] = nc_count
        info["device_memory_mb"] = total_memory / nc_count / (1024 ** 2) if nc_count > 0 else 0
        info["neuron_device_count"] = len(neuron_data)
        info["neuron_core_count"] = nc_count

        info["torch_version"] = torch.__version__
        try:
            info["torch_xla_version"] = importlib.metadata.version("torch-xla")
        except Exception:
            info["torch_xla_version"] = "unknown"
        try:
            info["neuronx_cc_version"] = importlib.metadata.version("neuronx-cc")
        except Exception:
            info["neuronx_cc_version"] = "unknown"
        try:
            info["torch_neuronx_version"] = importlib.metadata.version("torch-neuronx")
        except Exception:
            info["torch_neuronx_version"] = "unknown"

        return info

    def get_default_envs(self):
        return NEURON_ENV

    def get_provider_info(self):
        return NEURON_PROVIDER

    # ── device management ─────────────────────────────

    def get_torch_device_name(self):
        return "xla"

    def get_device_name(self, index=0):
        return self._get_instance_type()

    def get_device_properties(self, index=0):
        return {
            "name": self._get_instance_type(),
            "total_memory": self.backend_info.get("device_memory_mb", 0) * (1024 ** 2),
        }

    def get_mem_info(self, index=0):
        total = int(self.backend_info.get("device_memory_mb", 0) * (1024 ** 2))
        return (total, total)

    def get_device_count(self):
        count = self.backend_info.get("device_count", 0)
        return count, list(range(count))

    def set_device(self, device_index: int):
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(device_index)
        # Import torch_xla here to register the XLA backend with PyTorch
        # in this subprocess. Must happen after NEURON_RT_VISIBLE_CORES is
        # set and before any tensor operations (torch.empty(device="xla")).
        import torch_xla  # noqa: F401

    def get_device(self):
        import torch_xla.core.xla_model as xm
        return xm.xla_device()

    def device_synchronize(self):
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        xm.wait_device_ops()

    def empty_cache(self):
        pass

    # ── ccl related ───────────────────────────────────

    def get_dist_module(self):
        return dist

    def get_dist_backend(self):
        return "xla"

    def op_group_barrier(self, op_group=None, group_size=1):
        import torch_xla.core.xla_model as xm
        if dist.is_initialized() and group_size > 1:
            dist.all_reduce(
                torch.tensor([1], dtype=torch.int32, device=self.get_torch_device_name()),
                op=dist.ReduceOp.SUM,
                group=op_group
            )
            xm.mark_step()
            xm.wait_device_ops()

    # ── core_perf override ────────────────────────────

    def core_perf(
        self, op_instance,
        warmup_iterations, prefer_iterations,
        tensor_list,
        profiling=True
    ):
        import torch_xla.core.xla_model as xm

        op_group = op_instance.op_group
        group_size = op_instance.group_size

        self.op_group_barrier(op_group=op_group, group_size=group_size)
        self.device_synchronize()

        # Warmup -- extra iterations to absorb XLA compilation
        effective_warmup = max(warmup_iterations, 4)
        for i in range(effective_warmup):
            op_instance.core_run(tensor_list[i % len(tensor_list)])
            xm.mark_step()
        xm.wait_device_ops()

        # Timed iterations
        self.op_group_barrier(op_group=op_group, group_size=group_size)
        xm.wait_device_ops()

        start_time = time.perf_counter_ns()
        for i in range(prefer_iterations):
            op_instance.core_run(tensor_list[i % len(tensor_list)])
        xm.mark_step()
        xm.wait_device_ops()
        end_time = time.perf_counter_ns()

        latency_us = (end_time - start_time) / 1e3 / prefer_iterations
        return latency_us, []
