#include "kernels.hpp"
#include <HalideBuffer.h>
#include <HalideRuntime.h>
#include <cassert>
#include <chrono>
#include <lattice_symmetries/lattice_symmetries.h>
#include <random>

auto make_non_symmetric_basis() {
  ls_error_code status;
  ls_group *group = nullptr;
  status = ls_create_group(&group, 0, nullptr);
  assert(status == LS_SUCCESS);

  ls_spin_basis *basis = nullptr;
  status = ls_create_spin_basis(&basis, group, 24, 12, /*spin_inversion=*/0);
  assert(status == LS_SUCCESS);
  ls_destroy_group(group);

  return basis;
}

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

  halide_kernel(ls_group const *group)
      : halide_kernel{group, ls_get_group_size(group),
                      static_cast<unsigned>(
                          std::max(ls_group_get_network_depth(group), 0))} {}

  halide_kernel(ls_group const *group, unsigned const number_masks,
                unsigned const depth)
      : _masks(number_masks, depth), _eigvals_re(number_masks),
        _eigvals_im(number_masks),
        _shifts(depth), _x{}, _repr{}, _character{}, _norm{},
        _flip_mask{get_flip_mask_64(static_cast<unsigned>(
            std::max(ls_group_get_number_spins(group), 0)))},
        _kernel{get_general_kernel()} {
    fprintf(stderr, "_flip_mask = %zu\n", _flip_mask);
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

auto make_basis_4x6(bool const symmetric, int const spin_inversion) {
  ls_error_code status;
  // clang-format off
  unsigned const sites[] =
    { 0,  1,  2,  3,  4,  5,
      6,  7,  8,  9, 10, 11,
     12, 13, 14, 15, 16, 17,
     18, 19, 20, 21, 22, 23};
  unsigned const T_x[] =
    { 1,  2,  3,  4,  5,  0,
      7,  8,  9, 10, 11,  6,
     13, 14, 15, 16, 17, 12,
     19, 20, 21, 22, 23, 18};
  unsigned const T_y[] =
    { 6,  7,  8,  9, 10, 11,
     12, 13, 14, 15, 16, 17,
     18, 19, 20, 21, 22, 23,
      0,  1,  2,  3,  4,  5};
  unsigned const P_x[] =
    { 5,  4,  3,  2,  1,  0,
     11, 10,  9,  8,  7,  6,
     17, 16, 15, 14, 13, 12,
     23, 22, 21, 20, 19, 18};
  unsigned const P_y[] =
    {18, 19, 20, 21, 22, 23,
     12, 13, 14, 15, 16, 17,
      6,  7,  8,  9, 10, 11,
      0,  1,  2,  3,  4,  5};
  // clang-format on
  ls_symmetry *T_x_symmetry = nullptr;
  status = ls_create_symmetry(&T_x_symmetry, std::size(T_x), T_x, 0);
  assert(status == LS_SUCCESS);

  ls_symmetry *T_y_symmetry = nullptr;
  status = ls_create_symmetry(&T_y_symmetry, std::size(T_y), T_y, 0);
  assert(status == LS_SUCCESS);

  ls_symmetry *P_x_symmetry = nullptr;
  status = ls_create_symmetry(&P_x_symmetry, std::size(P_x), P_x, 0);
  assert(status == LS_SUCCESS);

  ls_symmetry *P_y_symmetry = nullptr;
  status = ls_create_symmetry(&P_y_symmetry, std::size(P_y), P_y, 0);
  assert(status == LS_SUCCESS);

  ls_symmetry const *generators[] = {T_x_symmetry, T_y_symmetry, P_x_symmetry,
                                     P_y_symmetry};
  ls_group *group = nullptr;
  if (symmetric) {
    status = ls_create_group(&group, std::size(generators), generators);
  } else {
    status = ls_create_group(&group, 0, nullptr);
  }
  assert(status == LS_SUCCESS);
  ls_destroy_symmetry(T_x_symmetry);
  ls_destroy_symmetry(T_y_symmetry);
  ls_destroy_symmetry(P_x_symmetry);
  ls_destroy_symmetry(P_y_symmetry);

  ls_spin_basis *basis = nullptr;
  status = ls_create_spin_basis(&basis, group, 24, 12,
                                symmetric ? spin_inversion : 0);
  assert(status == LS_SUCCESS);

  halide_kernel kernel{group};
  ls_destroy_group(group);
  return std::make_tuple(basis, std::move(kernel));
}

auto make_basis_5x5(bool symmetric) {
  ls_error_code status;
  // clang-format off
  unsigned const sites[] =
    { 0,  1,  2,  3,  4,
      5,  6,  7,  8,  9,
     10, 11, 12, 13, 14,
     15, 16, 17, 18, 19,
     20, 21, 22, 23, 24};
  unsigned const T_x[] =
    { 1,  2,  3,  4,  0,
      6,  7,  8,  9,  5,
     11, 12, 13, 14, 10,
     16, 17, 18, 19, 15,
     21, 22, 23, 24, 20};
  unsigned const T_y[] =
    { 5,  6,  7,  8,  9,
     10, 11, 12, 13, 14,
     15, 16, 17, 18, 19,
     20, 21, 22, 23, 24,
      0,  1,  2,  3,  4};
  unsigned const P_x[] =
    { 4,  3,  2,  1,  0,
      9,  8,  7,  6,  5,
     14, 13, 12, 11, 10,
     19, 18, 17, 16, 15,
     24, 23, 22, 21, 20};
  unsigned const P_y[] =
    {20, 21, 22, 23, 24,
     15, 16, 17, 18, 19,
     10, 11, 12, 13, 14,
      5,  6,  7,  8,  9,
      0,  1,  2,  3,  4};
  unsigned const R_90[] =
    { 4, 9, 14, 19, 24,
      3, 8, 13, 18, 23,
      2, 7, 12, 17, 22,
      1, 6, 11, 16, 21,
      0, 5, 10, 15, 20};
  // clang-format on
  ls_symmetry *T_x_symmetry = nullptr;
  status = ls_create_symmetry(&T_x_symmetry, std::size(T_x), T_x, 0);
  assert(status == LS_SUCCESS);

  ls_symmetry *T_y_symmetry = nullptr;
  status = ls_create_symmetry(&T_y_symmetry, std::size(T_y), T_y, 0);
  assert(status == LS_SUCCESS);

  ls_symmetry *P_x_symmetry = nullptr;
  status = ls_create_symmetry(&P_x_symmetry, std::size(P_x), P_x, 0);
  assert(status == LS_SUCCESS);

  ls_symmetry *P_y_symmetry = nullptr;
  status = ls_create_symmetry(&P_y_symmetry, std::size(P_y), P_y, 0);
  assert(status == LS_SUCCESS);

  ls_symmetry *R_90_symmetry = nullptr;
  status = ls_create_symmetry(&R_90_symmetry, std::size(R_90), R_90, 0);
  assert(status == LS_SUCCESS);

  ls_symmetry const *generators[] = {T_x_symmetry, T_y_symmetry, P_x_symmetry,
                                     P_y_symmetry, R_90_symmetry};
  ls_group *group = nullptr;
  if (symmetric) {
    status = ls_create_group(&group, std::size(generators), generators);
  } else {
    status = ls_create_group(&group, 0, nullptr);
  }
  assert(status == LS_SUCCESS);
  ls_destroy_symmetry(T_x_symmetry);
  ls_destroy_symmetry(T_y_symmetry);
  ls_destroy_symmetry(P_x_symmetry);
  ls_destroy_symmetry(P_y_symmetry);
  ls_destroy_symmetry(R_90_symmetry);

  ls_spin_basis *basis = nullptr;
  status = ls_create_spin_basis(&basis, group, 25, 13, /*spin_inversion=*/0);
  assert(status == LS_SUCCESS);

  halide_kernel kernel{group};

  ls_destroy_group(group);

  return std::make_tuple(basis, std::move(kernel));
}

int main() {
  // ls_enable_logging();
  auto [basis, kernel] = make_basis_4x6(true, 0);
  auto [full_basis, _] = make_basis_4x6(false, 0);
  ls_build(full_basis);
  ls_states *states;
  ls_get_states(&states, full_basis);
  std::vector<uint64_t> spins{ls_states_get_data(states),
                              ls_states_get_data(states) +
                                  ls_states_get_size(states)};
  ls_destroy_states(states);
  ls_destroy_spin_basis(full_basis);
  std::mt19937 generator(548291);
  std::shuffle(spins.begin(), spins.end(), generator);

  std::vector<uint64_t> counts;
  counts.reserve(30);
  counts.push_back(1);
  while (2 * counts.back() < spins.size()) {
    counts.push_back(2 * counts.back());
  }
  counts.push_back(spins.size());

  std::vector<uint64_t> repr_1(spins.size());
  std::vector<std::complex<double>> character_1(spins.size());
  std::vector<double> norm_1(spins.size());
  auto repr_2 = repr_1;
  auto character_2 = character_1;
  auto norm_2 = norm_1;

  for (auto const n : counts) {
    auto t1 = std::chrono::steady_clock::now();
    for (uint64_t i = 0U; i < n; ++i) {
      ls_get_state_info(basis,
                        reinterpret_cast<ls_bits512 const *>(spins.data() + i),
                        reinterpret_cast<ls_bits512 *>(repr_1.data() + i),
                        character_1.data() + i, norm_1.data() + i);
    }
    auto t2 = std::chrono::steady_clock::now();
    auto const old_time = std::chrono::duration<double>(t2 - t1).count();

    t1 = std::chrono::steady_clock::now();
    kernel(n, spins.data(), repr_2.data(), character_2.data(), norm_2.data());
    t2 = std::chrono::steady_clock::now();
    auto const new_time = std::chrono::duration<double>(t2 - t1).count();

    if (n == spins.size()) {
      for (uint64_t i = 0U; i < n; ++i) {
        if (repr_1[i] != repr_2[i] || character_1[i] != character_2[i] ||
            norm_1[i] != norm_2[i]) {
          fprintf(stderr, "Failed at i=%u\n", i);
          fprintf(stderr, "Repr: %zu vs %zu\n", repr_1[i], repr_2[i]);
          fprintf(stderr, "Character: %f vs %f\n", character_1[i].real(),
                  character_2[i].real());
          fprintf(stderr, "Norm: %f vs %f\n", norm_1[i], norm_2[i]);
          abort();
        }
      }
    }
    fprintf(stdout, "%zu\t%f\t%f\n", n, old_time, new_time);
  }

  ls_destroy_spin_basis(basis);
  return 0;
}
