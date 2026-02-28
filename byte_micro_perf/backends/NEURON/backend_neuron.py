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

    # ── xccl_infer_loop override ─────────────────────
    # Replace dist.all_gather_object (hangs on XLA backend) with
    # xm.rendezvous + pickle for Python object exchange across ranks.

    @staticmethod
    def _broadcast_object(obj, rank, world_size):
        """Broadcast a Python object from rank 0 to all ranks using XLA tensors."""
        import pickle
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

        if rank == 0:
            data = pickle.dumps(obj)
            size = len(data)
            # Broadcast size first
            size_tensor = torch.tensor([size], dtype=torch.int64, device=device)
            dist.broadcast(size_tensor, src=0)
            xm.mark_step()
            xm.wait_device_ops()
            # Broadcast data as byte tensor
            data_tensor = torch.tensor(list(data), dtype=torch.uint8, device=device)
            dist.broadcast(data_tensor, src=0)
            xm.mark_step()
            xm.wait_device_ops()
            return obj
        else:
            # Receive size
            size_tensor = torch.tensor([0], dtype=torch.int64, device=device)
            dist.broadcast(size_tensor, src=0)
            xm.mark_step()
            xm.wait_device_ops()
            size = size_tensor.item()
            # Receive data
            data_tensor = torch.zeros(size, dtype=torch.uint8, device=device)
            dist.broadcast(data_tensor, src=0)
            xm.mark_step()
            xm.wait_device_ops()
            data = bytes(data_tensor.cpu().tolist())
            return pickle.loads(data)

    @staticmethod
    def _gather_objects(obj, rank, world_size):
        """Gather Python objects from all ranks to all ranks using XLA tensors."""
        import pickle
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        data = pickle.dumps(obj)
        size = len(data)

        # All-gather sizes
        size_tensor = torch.tensor([size], dtype=torch.int64, device=device)
        all_sizes = xm.all_gather(size_tensor, dim=0)
        xm.mark_step()
        xm.wait_device_ops()

        max_size = all_sizes.max().item()

        # Pad data to max_size and all-gather
        padded = list(data) + [0] * (max_size - size)
        data_tensor = torch.tensor(padded, dtype=torch.uint8, device=device)
        all_data = xm.all_gather(data_tensor, dim=0)
        xm.mark_step()
        xm.wait_device_ops()

        # Unpack
        all_data_cpu = all_data.cpu()
        sizes_cpu = all_sizes.cpu().tolist()
        results = []
        offset = 0
        for s in sizes_cpu:
            s = int(s)
            chunk = bytes(all_data_cpu[offset:offset + s].tolist())
            results.append(pickle.loads(chunk))
            offset += max_size
        return results

    def xccl_infer_loop(
        self,
        local_process_rank, process_mapping,
        input_queue, output_queue,
        master_addr, xccl_port,
        node_world_size, node_rank,
    ):
        import signal
        import traceback
        import json
        import psutil
        import prettytable
        import torch_xla.core.xla_model as xm

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cur_process_info = process_mapping[local_process_rank]

        proc = psutil.Process(os.getpid())
        proc.cpu_affinity(cur_process_info["numa_cores"])

        local_device_id = cur_process_info["device_id"]
        self.set_device(local_device_id)

        local_world_size = len(process_mapping)
        local_rank = local_process_rank
        world_size = local_world_size * node_world_size
        rank = local_rank + node_rank * local_world_size

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(xccl_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        print(f"device_id: {local_device_id}, local: {local_rank}/{local_world_size}, world: {rank}/{world_size}")

        if world_size > 1:
            self.initialize_ccl(rank, world_size)
            # barrier via all_reduce (rendezvous hangs in subprocess context)
            self.op_group_barrier(op_group=None, group_size=world_size)

        if local_process_rank == 0:
            output_queue.put("success")

        dist_group_mapping = {}
        dist_group_mapping[world_size] = None

        while True:
            # rank 0 gets task from queue, broadcasts to all ranks
            if rank == 0:
                data = input_queue.get()
            else:
                data = None

            if world_size > 1:
                data = self._broadcast_object(data, rank, world_size)

            if data is None:
                break

            case_idx = data[0]
            op_name, op_provider, op_cls = data[1]
            task_case = data[2]

            result_dict = {}

            task_world_size = task_case.get("world_size", 1)
            if task_world_size > world_size:
                if rank == 0:
                    output_queue.put((case_idx, result_dict))
                continue

            if task_world_size > 1 and task_world_size not in dist_group_mapping:
                dist_group_mapping[task_world_size] = dist.new_group(ranks=list(range(task_world_size)))

            op_instance = None
            if rank < task_world_size:
                try:
                    op_instance = op_cls(
                        task_case, self,
                        op_group=dist_group_mapping.get(task_world_size, None),
                        group_size=task_world_size
                    )
                    op_instance.is_concurrent = True
                except Exception as e:
                    traceback.print_exc()

            # Check all ranks created op successfully
            if world_size > 1:
                check_results = self._gather_objects(
                    {"rank": rank, "result": op_instance is not None},
                    rank, world_size
                )
            else:
                check_results = [{"rank": rank, "result": op_instance is not None}]

            sorted_check = sorted(check_results, key=lambda x: x["rank"])[:task_world_size]
            if not all([x["result"] for x in sorted_check]):
                if rank == 0:
                    output_queue.put((case_idx, result_dict))
                continue

            target_dict = {}
            if rank < task_world_size:
                try:
                    target_dict = self.perf(op_instance, profiling=True)
                except Exception as e:
                    traceback.print_exc()

            # Gather results from all ranks
            if world_size > 1:
                result_exchange = self._gather_objects(
                    {"rank": rank, "device_id": local_device_id, "target_dict": target_dict},
                    rank, world_size
                )
            else:
                result_exchange = [{"rank": rank, "device_id": local_device_id, "target_dict": target_dict}]
            sorted_exchange_area = sorted(result_exchange, key=lambda x: x["rank"])[:task_world_size]

            target_dict_list = [x["target_dict"] for x in sorted_exchange_area]
            if not all(target_dict_list):
                if rank == 0:
                    output_queue.put((case_idx, result_dict))
                continue
            rank_list = [x["rank"] for x in sorted_exchange_area]
            device_id_list = [x["device_id"] for x in sorted_exchange_area]

            if rank == 0:
                new_target_dict = op_instance.merge_summary(target_dict_list)
                arguments_str = json.dumps(task_case)
                targets_str = json.dumps(new_target_dict, indent=4)

                pt = prettytable.PrettyTable()
                pt.field_names = ["key", "value"]
                pt.align = "l"
                pt.add_row(["op_name", op_name])
                pt.add_row(["op_provider", op_provider])
                pt.add_row(["rank", str(rank_list)])
                pt.add_row(["device_id", str(device_id_list)])
                pt.add_row(["idx", str(case_idx)])

                print(f"{pt}\n{arguments_str}\n{targets_str}\n")

                result_dict = new_target_dict
                output_queue.put((case_idx, result_dict))

        if world_size > 1:
            dist.destroy_process_group()

    # ── perf override (replace all_gather_object) ─────

    def perf(self, op_instance, profiling=True):
        import math
        import random
        import traceback
        import torch_xla.core.xla_model as xm

        tensor_size = op_instance.tensor_size
        device_mem_info = self.get_mem_info()
        avail_memory = device_mem_info[0]

        assume_avail_bytes = int(avail_memory * 0.9)
        assume_cache_size = 1 * (1024 ** 3)

        latency_us = 0.
        kernel_mapping = {}

        try:
            min_test_iters = 10
            sleep_time = 0.2
            max_test_time = 1e6
            max_data_cnt = 1
            if not op_instance.is_concurrent:
                if tensor_size > assume_avail_bytes:
                    raise RuntimeError("Not enough memory to run the op")
                elif 2 * tensor_size > assume_avail_bytes:
                    max_data_cnt = 1
                elif tensor_size > assume_cache_size:
                    max_data_cnt = 2
                else:
                    max_data_cnt = min(
                        math.floor(max(assume_avail_bytes, assume_cache_size) / tensor_size),
                        math.floor(assume_cache_size / tensor_size)
                    )

            tensor_list = op_instance.create_tensors(max_data_cnt)
            random.shuffle(tensor_list)

            latency_us, _ = self.core_perf(op_instance, 2, 2, tensor_list, profiling=False)
            prefer_iters = min(max(int(max_test_time / latency_us), 2), min_test_iters)
            if op_instance.group_size > 1:
                # Replace all_gather_object with XLA-compatible gather
                gathered = self._gather_objects(
                    {"rank": dist.get_rank(), "prefer_iters": prefer_iters},
                    dist.get_rank(), op_instance.group_size
                )
                prefer_iters = max(x["prefer_iters"] for x in gathered)
            time.sleep(sleep_time)
            latency_us, kernel_mapping = self.core_perf(op_instance, 2, prefer_iters, tensor_list, profiling=profiling)

            del tensor_list
            self.empty_cache()
        except Exception as e:
            traceback.print_exc()

        return op_instance.summary(latency_us, kernel_mapping)

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
