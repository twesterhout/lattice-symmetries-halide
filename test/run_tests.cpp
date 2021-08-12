#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "benes_network_general.h"
#include "benes_network_symmetric.h"
#include <HalideBuffer.h>
#include <complex>
#include <doctest.h>
#include <memory>

using namespace Halide::Runtime;

#if 0
TEST_CASE("benes_network") {

  uint64_t x = 0b0001000100;
  Buffer<uint64_t> masks(7);
  masks(0) = 0;
  masks(1) = 0;
  masks(2) = 0;
  masks(3) = 1;
  masks(4) = 1;
  masks(5) = 17;
  masks(6) = 341;
  Buffer<int> shifts(7);
  shifts(0) = 1;
  shifts(1) = 2;
  shifts(2) = 4;
  shifts(3) = 8;
  shifts(4) = 4;
  shifts(5) = 2;
  shifts(6) = 1;

  uint64_t r;
  auto r_buffer = Buffer<uint64_t>::make_scalar(&r);

  benes_network(x, masks, shifts, r_buffer);
  printf("%lu\n", r_buffer.number_of_elements());
  printf("%lu, %lu\n", x, r);
}
#endif

#if 1
TEST_CASE("4-spin chain (no symmetries)") {
  // 4 networks of depth 3
  uint64_t masks = 0;
  double eigvals = 0;
  int shifts[3] = {1, 2, 1};
  Buffer<uint64_t> masks_buffer{&masks, 0, 3};
  masks_buffer.transpose(0, 1);
  Buffer<double> eigvals_re_buffer{&eigvals, 0};
  Buffer<double> eigvals_im_buffer{&eigvals, 0};
  Buffer<int> shifts_buffer{&shifts[0], 3};

  uint64_t repr[1];
  std::complex<double> character[1];
  double norm[1];

  for (auto i = 0; i < 1; ++i) {
    repr[i] = std::numeric_limits<uint64_t>::max();
  }

  Buffer<uint64_t> repr_buffer{&repr[0], 1};
  Buffer<double> character_buffer{reinterpret_cast<double *>(&character), 2, 1};
  character_buffer.transpose(0, 1);
  Buffer<double> norm_buffer{&norm[0], 1};

  uint64_t x[1] = {0b0110};
  Buffer<uint64_t> x_buffer{&x[0], 1};
  benes_network_general(x_buffer, uint64_t{0b1111}, masks_buffer,
                        eigvals_re_buffer, eigvals_im_buffer, shifts_buffer,
                        repr_buffer, character_buffer, norm_buffer);

  for (auto i = 0; i < 1; ++i) {
    fprintf(stderr, "%lu -> %lu, %f + %fi, %f\n", x[i], repr[i],
            character[i].real(), character[i].imag(), norm[i]);
  }
}
#endif

#if 0
TEST_CASE("3-spin chain (no symmetries)") {
  // 4 networks of depth 3
  uint64_t masks[3][1] = {{0}, {0}, {0}};
  double eigvals_re[1] = {1};
  double eigvals_im[1] = {0};
  int shifts[3] = {1, 2, 1};

  Buffer<uint64_t> masks_buffer{&masks[0][0], 0, 3};
  masks_buffer.transpose(0, 1);
  Buffer<double> eigvals_re_buffer{&eigvals_re[0], 0};
  Buffer<double> eigvals_im_buffer{&eigvals_im[0], 0};
  Buffer<int> shifts_buffer{&shifts[0], 3};

  uint64_t repr[1];
  std::complex<double> character[1];
  double norm[1];

  Buffer<uint64_t> repr_buffer{&repr[0], 1};
  Buffer<double> character_buffer{reinterpret_cast<double *>(&character), 2, 1};
  character_buffer.transpose(0, 1);
  Buffer<double> norm_buffer{&norm[0], 1};

  uint64_t x[1] = {0b010};
  Buffer<uint64_t> x_buffer{&x[0], 1};
  benes_network(x_buffer, uint64_t{0b111}, masks_buffer, eigvals_re_buffer,
                eigvals_im_buffer, shifts_buffer, repr_buffer, character_buffer,
                norm_buffer);

  {
    auto const i = 0;
    printf("%lu -> %lu, %f + %fi, %f\n", x[i], repr[i], character[i].real(),
           character[i].imag(), norm[i]);
  }
}
#endif

#if 0
TEST_CASE("3-spin chain (translation only)") {
  // 4 networks of depth 3
  uint64_t masks[3][3] = {{0, 4, 0}, {1, 2, 0}, {1, 5, 0}};
  double eigvals_re[4] = {1, 1, 1};
  double eigvals_im[4] = {0, 0, 0};
  int shifts[3] = {1, 2, 1};

  Buffer<uint64_t> masks_buffer{&masks[0][0], 3, 3};
  masks_buffer.transpose(0, 1);
  Buffer<double> eigvals_re_buffer{&eigvals_re[0], 3};
  Buffer<double> eigvals_im_buffer{&eigvals_im[0], 3};
  Buffer<int> shifts_buffer{&shifts[0], 3};

  uint64_t repr[8];
  std::complex<double> character[8];
  double norm[8];

  Buffer<uint64_t> repr_buffer{&repr[0], 8};
  Buffer<double> character_buffer{reinterpret_cast<double *>(&character), 2, 8};
  character_buffer.transpose(0, 1);
  Buffer<double> norm_buffer{&norm[0], 8};

  uint64_t x[16] = {0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111};
  Buffer<uint64_t> x_buffer{&x[0], 8};
  benes_network(x_buffer, uint64_t{0b111}, masks_buffer, eigvals_re_buffer,
                eigvals_im_buffer, shifts_buffer, repr_buffer, character_buffer,
                norm_buffer);

  for (auto i = 0; i < 8; ++i) {
    printf("%lu -> %lu, %f + %fi, %f\n", x[i], repr[i], character[i].real(),
           character[i].imag(), norm[i]);
  }
}
#endif

#if 0
TEST_CASE("4-spin chain (translation only)") {
  // 4 networks of depth 3
  uint64_t masks[3][4] = {{0, 0, 0, 0}, {1, 3, 2, 0}, {5, 0, 5, 0}};
  double eigvals_re[4] = {1, 1, 1, 1};
  double eigvals_im[4] = {0, 0, 0, 0};
  int shifts[3] = {1, 2, 1};

  Buffer<uint64_t> masks_buffer{&masks[0][0], 4, 3};
  masks_buffer.transpose(0, 1);
  Buffer<double> eigvals_re_buffer{&eigvals_re[0], 4};
  Buffer<double> eigvals_im_buffer{&eigvals_im[0], 4};
  Buffer<int> shifts_buffer{&shifts[0], 3};

  uint64_t repr[16];
  std::complex<double> character[16];
  double norm[16];

  Buffer<uint64_t> repr_buffer{&repr[0], 16};
  Buffer<double> character_buffer{reinterpret_cast<double *>(&character), 2,
                                  16};
  character_buffer.transpose(0, 1);
  Buffer<double> norm_buffer{&norm[0], 16};

  uint64_t x[16] = {0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101,
                    0b0110, 0b0111, 0b1000, 0b1001, 0b1010, 0b1011,
                    0b1100, 0b1101, 0b1110, 0b1111};
  Buffer<uint64_t> x_buffer{&x[0], 16};
  benes_network(x_buffer, uint64_t{0b1111}, masks_buffer, eigvals_re_buffer,
                eigvals_im_buffer, shifts_buffer, repr_buffer, character_buffer,
                norm_buffer);

  for (auto i = 0; i < 16; ++i) {
    printf("%lu -> %lu, %f + %fi, %f\n", x[i], repr[i], character[i].real(),
           character[i].imag(), norm[i]);
  }
}
#endif
