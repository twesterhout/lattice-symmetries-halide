#include "kernels.hpp"
#include <HalideBuffer.h>
#include <HalideRuntime.h>
//
// Architecture-specific kernels
#ifdef __x86_64__
#include "ls_internal_state_info_general_kernel_64_avx.h"
#include "ls_internal_state_info_general_kernel_64_avx2.h"
#include "ls_internal_state_info_general_kernel_64_sse41.h"

#include "ls_internal_state_info_symmetric_kernel_64_avx.h"
#include "ls_internal_state_info_symmetric_kernel_64_avx2.h"
#include "ls_internal_state_info_symmetric_kernel_64_sse41.h"

#include "ls_internal_state_info_antisymmetric_kernel_64_avx.h"
#include "ls_internal_state_info_antisymmetric_kernel_64_avx2.h"
#include "ls_internal_state_info_antisymmetric_kernel_64_sse41.h"
#endif
// Generic implementation
#include "ls_internal_state_info_antisymmetric_kernel_64.h"
#include "ls_internal_state_info_general_kernel_64.h"
#include "ls_internal_state_info_symmetric_kernel_64.h"

namespace lattice_symmetries {

typedef int (*ls_internal_state_info_general_kernel_t)(
    struct halide_buffer_t *_x_buffer, uint64_t _flip_mask,
    struct halide_buffer_t *_masks_buffer,
    struct halide_buffer_t *_eigvals_re_buffer,
    struct halide_buffer_t *_eigvals_im_buffer,
    struct halide_buffer_t *_shifts_buffer,
    struct halide_buffer_t *_representative_buffer,
    struct halide_buffer_t *_character_buffer,
    struct halide_buffer_t *_norm_buffer);

struct halide_kernels_list_t {
  ls_internal_state_info_general_kernel_t general;
  ls_internal_state_info_general_kernel_t symmetric;
  ls_internal_state_info_general_kernel_t antisymmetric;
};

namespace {
halide_kernels_list_t init_halide_kernels() {
  __builtin_cpu_init();
#ifdef __x86_64__
  if (__builtin_cpu_supports("avx2") > 0) {
    return {&ls_internal_state_info_general_kernel_64_avx2,
            &ls_internal_state_info_symmetric_kernel_64_avx2,
            &ls_internal_state_info_antisymmetric_kernel_64_avx2};
  }
  if (__builtin_cpu_supports("avx") > 0) {
    return {&ls_internal_state_info_general_kernel_64_avx,
            &ls_internal_state_info_symmetric_kernel_64_avx,
            &ls_internal_state_info_antisymmetric_kernel_64_avx};
  }
  if (__builtin_cpu_supports("sse4.1") > 0) {
    return {&ls_internal_state_info_general_kernel_64_sse41,
            &ls_internal_state_info_symmetric_kernel_64_sse41,
            &ls_internal_state_info_antisymmetric_kernel_64_sse41};
  }
#endif
  return {&ls_internal_state_info_general_kernel_64,
          &ls_internal_state_info_symmetric_kernel_64,
          &ls_internal_state_info_antisymmetric_kernel_64};
}
} // namespace

struct halide_kernel {
  Halide::Runtime::Buffer<uint64_t, 2> _masks;
  Halide::Runtime::Buffer<double, 1> _eigvals_re;
  Halide::Runtime::Buffer<double, 1> _eigvals_im;
  Halide::Runtime::Buffer<unsigned, 1> _shifts;

  Halide::Runtime::Buffer<uint64_t, 1> _x;
  Halide::Runtime::Buffer<uint64_t, 1> _repr;
  Halide::Runtime::Buffer<double, 2> _character;
  Halide::Runtime::Buffer<double, 1> _norm;
  uint64_t _flip_mask;

  ls_internal_state_info_general_kernel_t _kernel;

  static auto get_flip_mask_64(unsigned const n) noexcept -> uint64_t {
    return n == 0U ? uint64_t{0} : ((~uint64_t{0}) >> (64U - n));
  }

public:
  halide_kernel(ls_group const *group, int spin_inversion)
      : halide_kernel{group, spin_inversion, ls_get_group_size(group),
                      static_cast<unsigned>(
                          std::max(ls_group_get_network_depth(group), 0))} {}

private:
  halide_kernel(ls_group const *group, int spin_inversion,
                unsigned const number_masks, unsigned const depth)
      : _masks(number_masks, depth), _eigvals_re(number_masks),
        _eigvals_im(number_masks),
        _shifts(depth), _x{}, _repr{}, _character{}, _norm{},
        _flip_mask{get_flip_mask_64(static_cast<unsigned>(
            std::max(ls_group_get_number_spins(group), 0)))},
        _kernel{} {
    auto kernels_list = init_halide_kernels();
    if (spin_inversion == 0) {
      _kernel = kernels_list.general;
    } else if (spin_inversion == 1) {
      _kernel = kernels_list.symmetric;
    } else if (spin_inversion == -1) {
      _kernel = kernels_list.antisymmetric;
    } else {
      assert(false);
    }

    _masks.transpose(0, 1);
    std::vector<std::complex<double>> temp(number_masks);
    ls_group_dump_symmetry_info(group, _masks.begin(), _shifts.begin(),
                                temp.data());
    for (auto i = 0U; i < number_masks; ++i) {
      _eigvals_re(i) = temp[i].real();
      _eigvals_im(i) = temp[i].imag();
    }

    _x.raw_buffer()->dimensions = 1;
    _x.raw_buffer()->dim[0] =
        halide_dimension_t{/*min=*/0, /*extent=*/0, /*stride=*/1,
                           /*flags=*/0};
    _repr.raw_buffer()->dimensions = 1;
    _repr.raw_buffer()->dim[0] =
        halide_dimension_t{/*min=*/0, /*extent=*/0, /*stride=*/1,
                           /*flags=*/0};
    _character.raw_buffer()->dimensions = 2;
    _character.raw_buffer()->dim[1] =
        halide_dimension_t{/*min=*/0, /*extent=*/2, /*stride=*/1,
                           /*flags=*/0};
    _character.raw_buffer()->dim[0] =
        halide_dimension_t{/*min=*/0, /*extent=*/0, /*stride=*/2,
                           /*flags=*/0};
    _norm.raw_buffer()->dimensions = 1;
    _norm.raw_buffer()->dim[0] =
        halide_dimension_t{/*min=*/0, /*extent=*/0, /*stride=*/1,
                           /*flags=*/0};
  }

public:
  auto operator()(uint64_t const count, uint64_t const *x, uint64_t *repr,
                  std::complex<double> *character, double *norm) {
    _x.raw_buffer()->host =
        reinterpret_cast<uint8_t *>(const_cast<uint64_t *>(x));
    _x.raw_buffer()->dim[0].extent = count;
    _repr.raw_buffer()->host =
        reinterpret_cast<uint8_t *>(const_cast<uint64_t *>(repr));
    _repr.raw_buffer()->dim[0].extent = count;
    _character.raw_buffer()->host = reinterpret_cast<uint8_t *>(
        const_cast<std::complex<double> *>(character));
    _character.raw_buffer()->dim[0].extent = count;
    _norm.raw_buffer()->host =
        reinterpret_cast<uint8_t *>(const_cast<double *>(norm));
    _norm.raw_buffer()->dim[0].extent = count;
    (*_kernel)(_x, _flip_mask, _masks, _eigvals_re, _eigvals_im, _shifts, _repr,
               _character, _norm);
  }
};

auto make_state_info_kernel(ls_group const *group, int spin_inversion)
    -> std::function<void(
        uint64_t const /*count*/, uint64_t const * /*x*/, uint64_t * /*repr*/,
        std::complex<double> * /*character*/, double * /*norm*/)> {
  return [f = halide_kernel{group, spin_inversion}](
             uint64_t const count, uint64_t const *x, uint64_t *repr,
             std::complex<double> *character,
             double *norm) mutable { f(count, x, repr, character, norm); };
}

} // namespace lattice_symmetries
