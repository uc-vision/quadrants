import pytest

import quadrants as qd
from quadrants.lang._template_mapper import TemplateMapper
from quadrants.lang.kernel_arguments import ArgMetadata

from tests import test_utils

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.needs_torch


@test_utils.test(arch=qd.cpu)
def test_direct_torch_tensors_reuse_semantic_specialization_cache() -> None:
    mapper = TemplateMapper([ArgMetadata(qd.types.ndarray(), "value")], [])
    first = torch.empty((2, 3), dtype=torch.float32)
    second = torch.empty((5, 7), dtype=torch.float32)

    assert mapper.lookup(False, (first,)) == mapper.lookup(False, (second,))
    assert len(mapper._mapping_cache) == 1
    assert len(mapper._mapping_cache_tracker) == 1


@test_utils.test(arch=qd.cuda)
def test_direct_cuda_tensors_reuse_specialization_with_live_launch_bindings() -> None:
    @qd.kernel
    def scale(source: qd.types.NDArray, target: qd.types.NDArray):
        for row_index, column_index in source:
            target[row_index, column_index] = 3 * source[row_index, column_index] + 1

    shapes = ((2, 3), (5, 7))
    sources = [
        torch.arange(rows * columns, dtype=torch.float32, device="cuda").reshape(rows, columns) + rows
        for rows, columns in shapes
    ]
    targets = [torch.empty_like(source) for source in sources]

    assert sources[0].data_ptr() != sources[1].data_ptr()
    assert targets[0].data_ptr() != targets[1].data_ptr()
    for source, target in zip(sources, targets):
        scale(source, target)
        torch.testing.assert_close(target, 3 * source + 1)

    mapper = scale._primal.mapper
    assert len(mapper.mapping) == 1
    assert len(mapper._mapping_cache) == 1
    assert len(mapper._mapping_cache_tracker) == 1

    noncontiguous = sources[1].T
    contiguous_target = torch.empty(noncontiguous.shape, dtype=noncontiguous.dtype, device=noncontiguous.device)
    with pytest.raises(ValueError, match="Non contiguous tensors are not supported"):
        scale(noncontiguous, contiguous_target)
    assert len(mapper._mapping_cache) == 1


@test_utils.test(arch=qd.cpu)
def test_direct_torch_tensor_mutations_invalidate_specialization_cache() -> None:
    mapper = TemplateMapper([ArgMetadata(qd.types.ndarray(), "value")], [])
    value = torch.empty((2, 3), dtype=torch.float32)

    specialization_ids = [mapper.lookup(False, (value,))[0]]
    value.resize_(6)
    specialization_ids.append(mapper.lookup(False, (value,))[0])
    value.data = torch.empty(6, dtype=torch.float64)
    specialization_ids.append(mapper.lookup(False, (value,))[0])
    value.requires_grad_(True)
    specialization_ids.append(mapper.lookup(False, (value,))[0])

    assert specialization_ids == [0, 1, 2, 3]
    assert len(mapper._mapping_cache) == 4


@test_utils.test(arch=qd.cpu)
def test_direct_torch_tensor_element_shape_mutation_is_revalidated() -> None:
    vector = qd.types.vector(n=3, dtype=qd.f32)
    mapper = TemplateMapper([ArgMetadata(qd.types.ndarray(dtype=vector), "value")], [])
    value = torch.empty((4, 3), dtype=torch.float32)

    assert mapper.lookup(False, (value,))[0] == 0
    value.resize_(4, 5)
    with pytest.raises(ValueError, match="required element_shape"):
        mapper.lookup(False, (value,))


@test_utils.test(arch=qd.cpu)
def test_torch_tensor_subclasses_keep_identity_cache_semantics() -> None:
    class TensorSubclass(torch.Tensor):
        pass

    mapper = TemplateMapper([ArgMetadata(qd.types.ndarray(), "value")], [])
    first = TensorSubclass((2, 3))
    second = TensorSubclass((2, 3))

    assert mapper.lookup(False, (first,))[0] == mapper.lookup(False, (second,))[0]
    assert len(mapper._mapping_cache) == 2
    assert len(mapper._mapping_cache_tracker) == 2
