#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "bayesH_core.hpp"

namespace py = pybind11;

py::array_t<double> bayesH_samples_py(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                      unsigned int n_iter,
                                      unsigned int seed) {
    auto buf = x.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input x must be 1D array.");
    }
    const size_t n = (size_t)buf.shape[0];
    const double* ptr = (const double*)buf.ptr;

    arma::vec data((arma::uword)n);
    for (size_t i = 0; i < n; ++i) data((arma::uword)i) = ptr[i];

    arma::vec samples = bayesH_samples_core(data, n_iter, seed);

    py::array_t<double> out(samples.n_elem);
    auto outbuf = out.request();
    double* outptr = (double*)outbuf.ptr;
    for (arma::uword i = 0; i < samples.n_elem; ++i) outptr[i] = samples(i);

    return out;
}

PYBIND11_MODULE(bayesH_py, m) {
    m.doc() = "HKp Bayesian Hurst exponent sampler (posterior samples)";
    m.def("bayesH_samples",
          &bayesH_samples_py,
          py::arg("x"),
          py::arg("n_iter") = 2000,
          py::arg("seed") = 42,
          "Return posterior samples of H (length n_iter).");
}