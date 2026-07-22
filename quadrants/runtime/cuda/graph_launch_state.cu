// Publishes one queued CUDA graph launch's mutable state from kernel parameters
// into the persistent device storage read by the graph's work kernels.
//
// After editing, regenerate the pre-built fatbin:
//
//   python scripts/build_graph_launch_state_fatbin.py

#include "graph_launch_state.h"

namespace quadrants::lang::cuda {

extern "C" __global__ void _qd_graph_launch_state_upload(GraphLaunchState state) {
  const auto index = threadIdx.x;
  if (index < state.count) {
    auto *destination = reinterpret_cast<std::uint64_t *>(state.writes[index].address);
    *destination = state.writes[index].value;
  }
}

}  // namespace quadrants::lang::cuda
