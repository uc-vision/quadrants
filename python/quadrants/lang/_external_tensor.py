from dataclasses import dataclass

try:
    import torch

    TORCH_TENSOR_TYPE = torch.Tensor
except ImportError:
    TORCH_TENSOR_TYPE = None


@dataclass(frozen=True, slots=True)
class ExternalTensorSpecializationSlot:
    """Compile-specialization fields for one direct external tensor argument.

    Runtime tensor state such as storage, pointers, sizes, devices, and contiguity deliberately does not belong here.
    """

    argument_index: int
    element_dimensions: int
    infer_grad_requirement: bool
