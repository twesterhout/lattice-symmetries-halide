#include "kernels.hpp"
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

template <bool Symmetric> auto make_basis_4x6(int const spin_inversion) {
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
  if (Symmetric) {
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
                                Symmetric ? spin_inversion : 0);
  assert(status == LS_SUCCESS);

  if constexpr (Symmetric) {
    auto kernel =
        lattice_symmetries::make_state_info_kernel(group, spin_inversion);
    ls_destroy_group(group);
    return std::make_tuple(basis, std::move(kernel));
  } else {
    return basis;
  }
  // auto kernel =
  //     lattice_symmetries::make_state_info_kernel(group, spin_inversion);
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

  auto kernel =
      lattice_symmetries::make_state_info_kernel(group, /*spin_inversion=*/0);
  ls_destroy_group(group);

  return std::make_tuple(basis, std::move(kernel));
}

int main() {
  ls_enable_logging();
  auto [basis, kernel] = make_basis_4x6<true>(1);
  auto full_basis = make_basis_4x6<false>(0);
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
