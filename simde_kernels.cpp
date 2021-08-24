#include <cassert>
#include <cstdint>
#include <utility>
#include <x86/avx2.h>

namespace lattice_symmetries {

namespace {

//
//        spin
//    configuration
//         |
//         v
// x_ptr   A0 B0 C0 D0
//         A1 B1 C1 D1
//         ...
//         A7 B7 C7 D7
//
// out_ptr A0 B0 C0 D0
//         A1 B1 C1 D1
//         ...
//         A7 B7 C7 D7
//
// m_ptr   M0 M1 M2 M3 M4 M5 M6 M7
//
//
template <int Shift>
auto bit_permute_step_4x8(simde__m256i const *__restrict__ x_ptr,
                          uint64_t const *__restrict__ m_ptr,
                          simde__m256i *__restrict__ out_ptr) noexcept -> void {
  constexpr auto vector_size = 4;
  static_assert(vector_size * sizeof(uint64_t) == sizeof(simde__m256i));
  for (auto i = 0; i < 8; ++i, ++x_ptr, ++m_ptr, ++out_ptr) {
    auto const x = simde_mm256_load_si256(x_ptr);
    auto const m = simde_mm256_set1_epi64x(static_cast<int64_t>(*m_ptr));
    simde__m256i y;
    // y <- x ^ (x >> Shift)
    if constexpr (Shift % 64 == 0) {
      if (i + (Shift / 64) < 8) {
        auto const t = simde_mm256_load_si256(x_ptr + (Shift / 64));
        y = simde_mm256_xor_si256(x, t);
      } else {
        y = x;
      }
    } else {
      static_assert(0 < Shift && Shift < 64);
      if (i == 7) { // There is no i + 1
        y = simde_mm256_srli_epi64(x, Shift);
      } else {
        auto const t = simde_mm256_load_si256(x_ptr + 1);
        y = simde_mm256_or_si256(simde_mm256_slli_epi64(t, 64 - Shift),
                                 simde_mm256_srli_epi64(x, Shift));
      }
      y = simde_mm256_xor_si256(x, y);
    }
    // y <- y & m
    y = simde_mm256_and_si256(y, m);
    // y <- y ^ (y << Shift)
    if constexpr (Shift % 64 == 0) {
      if (i - (Shift / 64) >= 0) {
        auto const t = simde_mm256_load_si256(x_ptr - (Shift / 64));
        y = simde_mm256_xor_si256(y, t);
      }
    } else {
      static_assert(0 < Shift && Shift < 64);
      if (i == 0) { // There is no i - 1
        y = simde_mm256_xor_si256(y, simde_mm256_slli_epi64(y, Shift));
      } else {
        auto t = simde_mm256_load_si256(x_ptr - 1);
        t = simde_mm256_or_si256(simde_mm256_srli_epi64(t, 64 - Shift),
                                 simde_mm256_slli_epi64(y, Shift));
        y = simde_mm256_xor_si256(y, t);
      }
    }
    simde_mm256_storeu_si256(out_ptr, y);
  }
}

auto benes_forward_kernel_4x8(simde__m256i const *__restrict__ x_ptr,
                              uint64_t const *__restrict__ m_ptr,
                              unsigned const depth,
                              simde__m256i *__restrict__ out_ptr) noexcept
    -> void {
  using kernel_type =
      void (*)(simde__m256i const *, uint64_t const *, simde__m256i *);
  constexpr kernel_type kernels[] = {
      &bit_permute_step_4x8<1>,   &bit_permute_step_4x8<2>,
      &bit_permute_step_4x8<4>,   &bit_permute_step_4x8<8>,
      &bit_permute_step_4x8<16>,  &bit_permute_step_4x8<32>,
      &bit_permute_step_4x8<64>,  &bit_permute_step_4x8<128>,
      &bit_permute_step_4x8<256>,
  };
  assert((depth + 1) / 2 <= std::size(kernels));
  alignas(32) uint64_t temp[8][4];

  if (depth == 0) {
    return;
  }

  kernels[0](x_ptr, m_ptr, out_ptr);
  if (depth == 1) {
    return;
  }

  auto i = 1;
  auto *input = out_ptr;
  auto *output = reinterpret_cast<simde__m256i *>(&temp[0][0]);
  for (; i < (depth - 1) / 2; ++i) {
    kernels[i](input, m_ptr + 8 * i, output);
    std::swap(output, input);
  }
  assert(i == (depth - 1) / 2);
  for (; i-- > 0;) {
    kernels[i](input, m_ptr + 8 * i, output);
    std::swap(output, input);
  }
}

auto transpose_kernel_4x4(simde__m256i _r0, simde__m256i _r1, simde__m256i _r2,
                          simde__m256i _r3, simde__m256i *output) noexcept
    -> void {
  // r0: {00, 01, 02, 03}
  // r1: {10, 11, 12, 13},
  // ...
  auto r0 = simde_mm256_castsi256_pd(_r0);
  auto r1 = simde_mm256_castsi256_pd(_r1);
  auto r2 = simde_mm256_castsi256_pd(_r2);
  auto r3 = simde_mm256_castsi256_pd(_r3);

  // a <- {20, 30, 23, 33}
  auto a = simde_mm256_shuffle_pd(r2, r3, 0b0011);
  // b <- {00, 10, 03, 13}
  auto b = simde_mm256_shuffle_pd(r0, r1, 0b0011);
  // c <- {21, 31, 22, 32}
  auto c = simde_mm256_shuffle_pd(r2, r3, 0b1100);
  // d <- {01, 11, 02, 12}
  auto d = simde_mm256_shuffle_pd(r0, r1, 0b1100);

  // r0 <- {00, 10, 20, 30}
  r0 = simde_mm256_permute2f128_pd(a, b, 0x2);
  // r1 <- {01, 11, 21, 31}
  r1 = simde_mm256_permute2f128_pd(c, d, 0x2);
  // r2 <- {02, 12, 22, 32}
  r2 = simde_mm256_permute2f128_pd(c, d, 0x13);
  // r2 <- {03, 13, 23, 33}
  r3 = simde_mm256_permute2f128_pd(a, b, 0x13);

  simde_mm256_store_si256(output, simde_mm256_castpd_si256(r0));
  simde_mm256_store_si256(output + 1, simde_mm256_castpd_si256(r1));
  simde_mm256_store_si256(output + 2, simde_mm256_castpd_si256(r2));
  simde_mm256_store_si256(output + 3, simde_mm256_castpd_si256(r3));
}

auto transpose_kernel_4x8(simde__m256i const *input,
                          simde__m256i *output) noexcept -> void {
  auto r0 = simde_mm256_load_si256(input);
  auto r1 = simde_mm256_load_si256(input + 2);
  auto r2 = simde_mm256_load_si256(input + 4);
  auto r3 = simde_mm256_load_si256(input + 6);
  transpose_kernel_4x4(r0, r1, r2, r3, output);
  r0 = simde_mm256_load_si256(input + 1);
  r1 = simde_mm256_load_si256(input + 3);
  r2 = simde_mm256_load_si256(input + 5);
  r3 = simde_mm256_load_si256(input + 7);
  transpose_kernel_4x4(r0, r1, r2, r3, output + 4);
}

auto simde_mm256_cmpgt_epu64(simde__m256i const a,
                             simde__m256i const b) noexcept -> simde__m256i {
  auto const sign64 = simde_mm256_set1_epi64x(0x8000000000000000);
  auto const aflip = simde_mm256_xor_si256(a, sign64);
  auto const bflip = simde_mm256_xor_si256(b, sign64);
  return simde_mm256_cmpgt_epi64(aflip, bflip);
}

#if 0
template <int spin_inversion>
auto state_info_512_kernel_4x8(unsigned number_masks, unsigned depth,
                               uint64_t const *masks_ptr,
                               simde__m256i const *x_ptr,
                               simde__m256i const *flip_mask_ptr,
                               simde__m256i *repr_ptr,
                               simde__m256d *character_ptr,
                               simde__m256d *norm_ptr) {

  alignas(32) uint64_t x_storage[8][4];
  alignas(32) uint64_t y_storage[8][4];
  alignas(32) uint64_t r_storage[8][4];

  // Load x
  transpose_kernel_4x8(x_ptr,
                       reinterpret_cast<simde__m256i *>(&x_storage[0][0]));

  //

  simde__m256i flip_mask;
  if constexpr (spin_inversion != 0) {
    flip_mask = simde_mm256_loadu_si256(flip_mask_ptr);
  }

  for (auto i = 0U; i < number_masks; ++i) {
    benes_forward_kernel_4x8(
        reinterpret_cast<simde__m256i const *>(&x_storage[0][0]),
        masks_ptr + 8 * depth * i, depth,
        reinterpret_cast<simde__m256i *>(&y_storage[0][0]));

    simde__m256i is_equal = simde_mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF);
    simde__m256i is_done = simde_mm256_setzero_si256();
    simde__m256i is_less = simde_mm256_setzero_si256();
    for (auto j = 0; j < 8; ++j) {
      auto y_j = simde_mm256_load_si256(
          reinterpret_cast<simde__m256i const *>(&y_storage[j][0]));
      auto x_j = simde_mm256_load_si256(
          reinterpret_cast<simde__m256i const *>(&x_storage[j][0]));
      auto const equal_j = simde_mm256_cmpeq_epi64(x_j, y_j);
      auto const less_j = simde_mm256_cmpgt_epu64(x_j, y_j);
      is_equal = simde_mm256_and_si256(is_equal, equal_j);
      is_less = simde_mm256_and_si256(
            is_less,
            simde_mm256_
    }

    if constexpr (spin_inversion == 0) {
    }
  }

  simde__m256i repr;
  simde__m256d character_re;
  simde__m256d character_im;
  simde__m256d norm;

  repr = simde_mm256_loadu_si256(x_ptr);
  character_re = simde_mm256_set1_pd(1.0);
  character_im = simde_mm256_setzero_pd();
  norm = simde_mm256_set1_pd(0.0);

#if 0
  if (number_masks == 0) {
    if constexpr (spin_inversion == 0) {
      norm = simde_mm256_set1_pd(1.0);
    } else {
      static_assert(spin_inversion == -1 || spin_inversion == 1);
      auto const flip_mask = simde_mm256_loadu_si256(flip_mask_ptr);
      auto const should_flip = if constexpr (spin_inversion == 1) {
        simde__m256i y = simde_mm256_xor_si256(repr, flip_mask);
      }
    }

    simde_mm256_storeu_si256(repr_ptr, simde_mm256_loadu_si256(x_ptr));
    auto const one = simde_mm256_set1_pd(1.0);
    auto const complex_one =
        simde_mm256_unpackhi_pd(one, simde_mm256_setzero_pd());
    simde_mm256_storeu_si256(character_ptr, complex_one);
    simde_mm256_storeu_si256(character_ptr + 1, complex_one);
    simde_mm256_storeu_si256(norm_ptr, simde_mm256_loadu_si256(one));
    return;
  }
#endif

  ls_bits512
      buffer; // NOLINT: buffer is initialized inside the loop before it is used
  auto r = bits;
  auto n = 0.0;
  auto e = std::complex<double>{1.0};

  for (auto const &symmetry : basis_body.symmetries) {
    buffer = bits;
    symmetry.network(buffer);
    if (buffer < r) {
      r = buffer;
      e = symmetry.eigenvalue;
    } else if (buffer == bits) {
      n += symmetry.eigenvalue.real();
    }
    if (basis_header.spin_inversion != 0) {
      buffer ^= flip_mask;
      if (buffer < r) {
        r = buffer;
        e = flip_coeff * symmetry.eigenvalue;
      } else if (buffer == bits) {
        n += flip_coeff * symmetry.eigenvalue.real();
      }
    }
  }

  // We need to detect the case when norm is not zero, but only because of
  // inaccurate arithmetics
  constexpr auto norm_threshold = 1.0e-5;
  if (std::abs(n) <= norm_threshold) {
    n = 0.0;
  }
  LATTICE_SYMMETRIES_ASSERT(n >= 0.0, "");
  auto const group_size =
      (static_cast<unsigned>(basis_header.spin_inversion != 0) + 1) *
      basis_body.symmetries.size();
  n = std::sqrt(n / static_cast<double>(group_size));

  // Save results
  representative = r;
  character = e;
  norm = n;
}
#endif

// We set 64-bit mask to
//
//     0000....0011....1111
//               ~~~~~~~~~~
//                   n
constexpr auto get_flip_mask_64(unsigned const n) noexcept -> uint64_t {
  // Play nice and do not shift by 64 bits
  return n == 0U ? uint64_t{0} : ((~uint64_t{0}) >> (64U - n));
}

auto get_flip_mask_512(unsigned n) noexcept -> simde__m256i {
  alignas(32) uint64_t mask[8];
  auto i = 0U;
  // NOLINTNEXTLINE: 64 is the number of bits in uint64_t
  for (; n >= 64U; ++i, n -= 64) {
    mask[i] = ~uint64_t{0};
  }
  if (n != 0U) {
    mask[i] = get_flip_mask_64(n);
    ++i;
  }
  for (; i < 8; ++i) {
    mask[i] = uint64_t{0};
  }
  return simde_mm256_load_si256(reinterpret_cast<simde__m256i const *>(mask));
}

} // namespace

} // namespace lattice_symmetries
