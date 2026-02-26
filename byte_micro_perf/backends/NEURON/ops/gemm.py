import sys
import pathlib

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.tensor_gemm_ops import GemmOp


class NeuronGemmOp(GemmOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        dtype = args_dict.get("dtype", "")
        if dtype == "tfloat32":
            raise NotImplementedError(
                "Neuron does not support tfloat32 (NVIDIA-specific)"
            )
        if dtype == "int8":
            raise NotImplementedError(
                "Neuron does not support int8 gemm via torch.matmul"
            )
        super().__init__(args_dict, backend, *args, **kwargs)


OP_MAPPING = {
    "torch": NeuronGemmOp
}
