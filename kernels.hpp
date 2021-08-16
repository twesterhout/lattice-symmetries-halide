#pragma once

#include <complex>
#include <cstdint>
#include <functional>
#include <lattice_symmetries/lattice_symmetries.h>

namespace lattice_symmetries {

auto make_state_info_kernel(ls_group const *group, int spin_inversion)
    -> std::function<void(
        uint64_t const /*count*/, uint64_t const * /*x*/, uint64_t * /*repr*/,
        std::complex<double> * /*character*/, double * /*norm*/)>;

} // namespace lattice_symmetries
