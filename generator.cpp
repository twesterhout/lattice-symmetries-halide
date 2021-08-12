#include "Halide.h"

using namespace Halide;

auto bit_permute_step_64(Expr x, Expr m, Expr d) -> Expr {
  auto y = ((x >> d) ^ x) & m;
  return (x ^ y) ^ (y << d);
}

class benes_network_generator
    : public Halide::Generator<benes_network_generator> {
public:
  GeneratorParam<int> _spin_inversion{"spin_inversion", /*default value*/ 0};

  Input<Buffer<uint64_t>> _x{"x", 1};
  Input<uint64_t> _flip_mask{"flip_mask"};
  Input<Buffer<uint64_t>> _masks{"masks", 2};
  Input<Buffer<double>> _eigvals_re{"eigvals_re", 1};
  Input<Buffer<double>> _eigvals_im{"eigvals_im", 1};
  Input<Buffer<unsigned>> _shifts{"shifts", 1};
  Output<Buffer<uint64_t>> _repr{"representative", 1};
  Output<Buffer<double>> _character{"character", 2};
  Output<Buffer<double>> _norm{"norm"};

  auto reduction_step_impl(Expr i, Tuple current, Expr y, Expr c_re,
                           Expr c_im) const -> Tuple {
    auto current_r = current[0];
    auto current_c_re = current[1];
    auto current_c_im = current[2];
    auto current_n = current[3];

    auto is_less = y < current_r;
    auto next_r = select(is_less, y, current_r);
    auto next_c_re = select(is_less, c_re, current_c_re);
    auto next_c_im = select(is_less, c_im, current_c_im);

    auto is_equal = y == _x(i);
    auto next_n = select(is_equal, current_n + c_re, current_n);

    return {next_r, next_c_re, next_c_im, next_n};
  }

  auto reduction_step(Expr i, Tuple current, Expr y, Expr c_re, Expr c_im) const
      -> Tuple {
    if (_spin_inversion == 0) {
      return reduction_step_impl(i, current, y, c_re, c_im);
    }
    if (_spin_inversion == 1) {
      auto temp = reduction_step_impl(i, current, y, c_re, c_im);
      return reduction_step_impl(i, temp, y ^ _flip_mask, c_re, c_im);
    }
    if (_spin_inversion == -1) {
      auto temp = reduction_step_impl(i, current, y, c_re, c_im);
      return reduction_step_impl(i, temp, y ^ _flip_mask, -c_re, -c_im);
    }
    throw std::runtime_error{"invalid spin_inversion"};
  }

  auto reduction_step(Tuple current, Tuple other) const -> Tuple {
    auto current_r = current[0];
    auto current_c_re = current[1];
    auto current_c_im = current[2];
    auto current_n = current[3];
    auto r = other[0];
    auto c_re = other[1];
    auto c_im = other[2];
    auto n = other[3];

    auto is_less = r < current_r;
    auto next_r = select(is_less, r, current_r);
    auto next_c_re = select(is_less, c_re, current_c_re);
    auto next_c_im = select(is_less, c_im, current_c_im);
    auto next_n = current_n + n;

    return {next_r, next_c_re, next_c_im, next_n};
  }

  void generate() {
    auto depth = _masks.dim(0).extent();
    auto number_masks = _masks.dim(1).extent();
    auto const chunk_size = 2 * natural_vector_size(type_of<uint64_t>());
    auto number_chunks = number_masks / chunk_size;
    auto number_rest = number_masks - number_chunks * chunk_size;

    Var i{"i"};

    // Apply masks to generate transformed spin configurations
    RDom k{0, depth, "k"};
    Func y_batched{"y_batched"};
    Var j_outer{"j_outer"}, j_inner{"j_inner"};
    y_batched(i, j_outer, j_inner) = _x(i);
    y_batched(i, j_outer, j_inner) = bit_permute_step_64(
        y_batched(i, j_outer, j_inner),
        _masks(k, j_outer * chunk_size + j_inner), _shifts(k));

    Func y_scalar{"y_scalar"};
    Var j_tail{"j_tail"};
    y_scalar(i, j_tail) = _x(i);
    y_scalar(i, j_tail) =
        bit_permute_step_64(y_scalar(i, j_tail), _masks(k, j_tail), _shifts(k));

    // Compute vectorized reduction
    RDom m_main{0, number_chunks, "m_main"};
    Func custom_reduction{"custom_reduction"};
    custom_reduction(i, j_inner) =
        Tuple{_x(i), cast<double>(1), cast<double>(0), cast<double>(0)};
    custom_reduction(i, j_inner) = reduction_step(
        i, custom_reduction(i, j_inner), y_batched(i, m_main, j_inner),
        _eigvals_re(m_main * chunk_size + j_inner),
        _eigvals_im(m_main * chunk_size + j_inner));

    // Reduce across one chunk
    RDom m_lane{1, chunk_size - 1, "m_lane"};
    Func reduction_lane{"reduction_lane"};
    reduction_lane(i) =
        Tuple{custom_reduction(i, 0)[0], custom_reduction(i, 0)[1],
              custom_reduction(i, 0)[2], custom_reduction(i, 0)[3]};
    reduction_lane(i) =
        reduction_step(reduction_lane(i), custom_reduction(i, m_lane));

    // Compute reduction over remaining elements
    Func reduction_scalar{"reduction_scalar"};
    RDom m_tail{number_chunks * chunk_size, number_rest, "m_tail"};
    reduction_scalar(i) = Tuple{reduction_lane(i)[0], reduction_lane(i)[1],
                                reduction_lane(i)[2], reduction_lane(i)[3]};
    reduction_scalar(i) =
        reduction_step(i, reduction_scalar(i), y_scalar(i, m_tail),
                       _eigvals_re(m_tail), _eigvals_im(m_tail));

    // Store results
    Var q;
    _repr(i) = reduction_scalar(i)[0];
    _character(i, q) = undef<double>();
    _character(i, 0) = reduction_scalar(i)[1];
    _character(i, 1) = reduction_scalar(i)[2];
    if (_spin_inversion == 0) {
      _norm(i) = sqrt(reduction_scalar(i)[3] / number_masks);
    } else {
      _norm(i) = sqrt(reduction_scalar(i)[3] / (2 * number_masks));
    }

    auto batch_size = _x.dim(0).extent();
    _x.dim(0).set_min(0).set_stride(1);
    _masks.dim(0).set_min(0).set_stride(number_masks);
    _masks.dim(1).set_min(0).set_stride(1);
    _eigvals_re.dim(0).set_min(0).set_stride(1).set_extent(number_masks);
    _eigvals_im.dim(0).set_min(0).set_stride(1).set_extent(number_masks);
    _shifts.dim(0).set_min(0).set_stride(1).set_extent(depth);
    _repr.dim(0).set_min(0).set_stride(1).set_extent(batch_size);
    _character.dim(0).set_min(0).set_stride(2).set_extent(batch_size);
    _character.dim(1).set_min(0).set_stride(1).set_extent(2);
    _norm.dim(0).set_min(0).set_stride(1).set_extent(batch_size);

    // Schedule
    y_batched.compute_at(custom_reduction, m_main);

    custom_reduction.compute_at(reduction_scalar, i);
    custom_reduction.vectorize(j_inner);
    custom_reduction.update(0).vectorize(j_inner);
    // reduction_lane.compute_at(reduction_scalar, i);
    reduction_scalar.compute_at(_repr, i);
    _character.update(0).compute_with(_repr, i);
    _character.update(1).compute_with(_repr, i);
    _norm.compute_with(_repr, i);
  }
};

HALIDE_REGISTER_GENERATOR(benes_network_generator, benes_network_generator)
