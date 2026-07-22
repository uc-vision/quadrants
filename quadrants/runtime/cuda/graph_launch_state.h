#pragma once

#include <cstddef>
#include <cstdint>

namespace quadrants::lang::cuda {

constexpr std::size_t kGraphLaunchWriteCapacity = 128;

struct GraphLaunchWrite {
  std::uint64_t address;
  std::uint64_t value;
};

struct GraphLaunchState {
  GraphLaunchWrite writes[kGraphLaunchWriteCapacity];
  std::uint32_t count;
};

static_assert(sizeof(GraphLaunchState) == 2056);

}  // namespace quadrants::lang::cuda
