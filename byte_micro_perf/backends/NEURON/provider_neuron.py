import importlib.metadata

NEURON_PROVIDER = {}

try:
    NEURON_PROVIDER["nki"] = {
        "neuronxcc": importlib.metadata.version("neuronxcc"),
    }
except Exception:
    pass
