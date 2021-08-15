#pragma once

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

// Architecture-specific kernels
#ifdef __x86_64__
#include "ls_internal_state_info_general_kernel_64_avx.h"
#include "ls_internal_state_info_general_kernel_64_avx2.h"
#include "ls_internal_state_info_general_kernel_64_sse41.h"
#endif
// Generic implementation
#include "ls_internal_state_info_general_kernel_64.h"

typedef int (*ls_internal_state_info_general_kernel_t)(
    struct halide_buffer_t *_x_buffer, uint64_t _flip_mask,
    struct halide_buffer_t *_masks_buffer,
    struct halide_buffer_t *_eigvals_re_buffer,
    struct halide_buffer_t *_eigvals_im_buffer,
    struct halide_buffer_t *_shifts_buffer,
    struct halide_buffer_t *_representative_buffer,
    struct halide_buffer_t *_character_buffer,
    struct halide_buffer_t *_norm_buffer);

static inline ls_internal_state_info_general_kernel_t get_general_kernel() {
  __builtin_cpu_init();
#ifdef __x86_64__
  if (__builtin_cpu_supports("avx2") > 0) {
    return &ls_internal_state_info_general_kernel_64_avx2;
  }
  if (__builtin_cpu_supports("avx") > 0) {
    return &ls_internal_state_info_general_kernel_64_avx;
  }
  if (__builtin_cpu_supports("sse4.1") > 0) {
    return &ls_internal_state_info_general_kernel_64_sse41;
  }
#endif
  return &ls_internal_state_info_general_kernel_64;
}
