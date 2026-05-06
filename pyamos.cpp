#include "amos.h"
#include "zbessel/zbessel.hh"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <vector>

namespace nb = nanobind;

NB_MODULE(pyamos, m) {
  m.def(
      "besh_xsf",
      [](std::complex<double> z, double fnu, int kode, int m, int n) {
        // Input validation
        if (fnu < 0.0) {
          throw std::invalid_argument("fnu must be non-negative");
        }

        if (n <= 0) {
          throw std::invalid_argument("n must be positive");
        }

        // Allocate output array using unique_ptr for automatic memory
        // management
        auto result = std::make_unique<std::complex<double>[]>(n);
        auto result_ptr = result.get();

        std::vector<std::complex<double>> cy(n + 16);
        std::complex<double> *cy_ptr = cy.data() + 8;

        int ierr = 0;
        // int nz = xsf::amos::besh(z, fnu, kode, m, n, result_ptr, &ierr);
        int nz = xsf::amos::besh(z, fnu, kode, m, n, cy_ptr, &ierr);

        if (ierr != 0)
          throw std::runtime_error("besh failed with ierr=" +
                                   std::to_string(ierr));

        // Copy results
        for (int j = 0; j < n; ++j) {
          result_ptr[j] = cy_ptr[j];
        }

        // check if guards were touched
        for (int j = 0; j < 8; ++j) {
          if (cy[j] != std::complex<double>(0.0, 0.0) ||
              cy[n + 8 + j] != std::complex<double>(0.0, 0.0)) {
            throw std::runtime_error("besh guard zones were modified");
          }
        }

        // Transfer ownership to numpy
        nb::capsule owner(result.release(), [](void *p) noexcept {
          delete[] static_cast<std::complex<double> *>(p);
        });

        size_t shape[1] = {static_cast<size_t>(n)};

        return nb::ndarray<nb::numpy, std::complex<double>>(result_ptr, 1,
                                                            shape, owner);
      },
      nb::arg("z"), nb::arg("fnu"), nb::arg("kode"), nb::arg("m"), nb::arg("n"),
      R"pbdoc(
Compute an n-member sequence of Hankel functions H(m, fnu + j - 1, z).

Parameters
----------
z : complex
    Complex argument.
fnu : float
    Order of the initial Hankel function (>= 0.0).
kode : int
    Scaling option (1 or 2):
    1 - returns the Hankel function as is
    2 - returns the Hankel function scaled by exp(-i*z*(3-2m))
m : int
    Kind of Hankel function (1 or 2).
n : int
    Number of members in the returned sequence.

Returns
-------
ndarray[complex128]
    1-D numpy array of length n containing H(m, fnu + j - 1, z) for j=1..n.

Raises
------
RuntimeError
    If the underlying routine reports an error (ierr != 0).
)pbdoc");
  m.def(
      "besh_zbessel",
      [](std::complex<double> z, double fnu, int kode, int m, int n) {
        // Input validation
        if (fnu < 0.0) {
          throw std::invalid_argument("fnu must be non-negative");
        }

        if (n <= 0) {
          throw std::invalid_argument("n must be positive");
        }

        // Allocate output array using unique_ptr for automatic memory
        // management
        auto result = std::make_unique<std::complex<double>[]>(n);
        auto result_ptr = result.get();

        // std::vector<double> cyr(n);
        // std::vector<double> cyi(n);

        std::vector<double> cyr(n + 16);
        std::vector<double> cyi(n + 16);
        double *cyr_ptr = cyr.data() + 8;
        double *cyi_ptr = cyi.data() + 8;

        int ierr = 0;

        int nz;
        // ierr = zbessel::zbesh(z.real(), z.imag(), fnu, kode, m, n,
        // cyr.data(),
        //                       cyi.data(), &nz);
        ierr = zbessel::zbesh(z.real(), z.imag(), fnu, kode, m, n, cyr_ptr,
                              cyi_ptr, &nz);

        if (ierr != 0)
          throw std::runtime_error("besh failed with ierr=" +
                                   std::to_string(ierr));

        // Copy to result_ptr
        // for (int j = 0; j < n; ++j) {
        //   result_ptr[j] = std::complex<double>(cyr[j], cyi[j]);
        // }

        // Copy results
        for (int j = 0; j < n; ++j) {
          result_ptr[j] = std::complex<double>(cyr_ptr[j], cyi_ptr[j]);
        }

        // check if guards were touched
        for (int j = 0; j < 8; ++j) {
          if (cyr[j] != 0.0 || cyi[j] != 0.0 || cyr[n + 8 + j] != 0.0 ||
              cyi[n + 8 + j] != 0.0) {
            throw std::runtime_error("besh guard zones were modified");
          }
        }

        // Transfer ownership to numpy
        nb::capsule owner(result.release(), [](void *p) noexcept {
          delete[] static_cast<std::complex<double> *>(p);
        });

        size_t shape[1] = {static_cast<size_t>(n)};

        return nb::ndarray<nb::numpy, std::complex<double>>(result_ptr, 1,
                                                            shape, owner);
      },
      nb::arg("z"), nb::arg("fnu"), nb::arg("kode"), nb::arg("m"), nb::arg("n"),
      R"pbdoc(
Compute an n-member sequence of Hankel functions H(m, fnu + j - 1, z).

Parameters
----------
z : complex
    Complex argument.
fnu : float
    Order of the initial Hankel function (>= 0.0).
kode : int
    Scaling option (1 or 2):
    1 - returns the Hankel function as is
    2 - returns the Hankel function scaled by exp(-i*z*(3-2m))
m : int
    Kind of Hankel function (1 or 2).
n : int
    Number of members in the returned sequence.

Returns
-------
ndarray[complex128]
    1-D numpy array of length n containing H(m, fnu + j - 1, z) for j=1..n.

Raises
------
RuntimeError
    If the underlying routine reports an error (ierr != 0).
)pbdoc");

  m.def(
      "besj_xsf",
      [](std::complex<double> z, double fnu, int kode, int n) {
        // Input validation
        if (fnu < 0.0) {
          throw std::invalid_argument("fnu must be non-negative");
        }

        if (n <= 0) {
          throw std::invalid_argument("n must be positive");
        }

        // Allocate output array using unique_ptr for automatic memory
        // management
        auto result = std::make_unique<std::complex<double>[]>(n);
        auto result_ptr = result.get();

        std::vector<std::complex<double>> cy(n + 16);
        std::complex<double> *cy_ptr = cy.data() + 8;

        int ierr = 0;
        int nz = xsf::amos::besj(z, fnu, kode, n, cy_ptr, &ierr);

        if (ierr != 0)
          throw std::runtime_error("besh failed with ierr=" +
                                   std::to_string(ierr));

        // Copy results
        for (int j = 0; j < n; ++j) {
          result_ptr[j] = cy_ptr[j];
        }

        // check if guards were touched
        for (int j = 0; j < 8; ++j) {
          if (cy[j] != std::complex<double>(0.0, 0.0) ||
              cy[n + 8 + j] != std::complex<double>(0.0, 0.0)) {
            throw std::runtime_error("besh guard zones were modified");
          }
        }

        // Transfer ownership to numpy
        nb::capsule owner(result.release(), [](void *p) noexcept {
          delete[] static_cast<std::complex<double> *>(p);
        });

        size_t shape[1] = {static_cast<size_t>(n)};

        return nb::ndarray<nb::numpy, std::complex<double>>(result_ptr, 1,
                                                            shape, owner);
      },
      nb::arg("z"), nb::arg("fnu"), nb::arg("kode"), nb::arg("n"),
      R"pbdoc(
Compute an n-member sequence of Bessel functions J(fnu + j - 1, z).

Parameters
----------
z : complex
    Complex argument.
fnu : float
    Order of the initial Bessel function (>= 0.0).
kode : int
    Scaling option (1 or 2):
    1 - returns the Bessel function as is
    2 - returns the Bessel function scaled by exp(-abs(z.imag))
n : int
    Number of members in the returned sequence.

Returns
-------
ndarray[complex128]
    1-D numpy array of length n containing J(fnu + j - 1, z) for j=1..n.

Raises
------
RuntimeError
    If the underlying routine reports an error (ierr != 0).
)pbdoc");

  m.def(
      "besj_zbessel",
      [](std::complex<double> z, double fnu, int kode, int n) {
        // Input validation
        if (fnu < 0.0) {
          throw std::invalid_argument("fnu must be non-negative");
        }

        if (n <= 0) {
          throw std::invalid_argument("n must be positive");
        }

        // Allocate output array using unique_ptr for automatic memory
        // management
        auto result = std::make_unique<std::complex<double>[]>(n);
        auto result_ptr = result.get();

        std::vector<double> cyr(n + 16);
        std::vector<double> cyi(n + 16);
        double *cyr_ptr = cyr.data() + 8;
        double *cyi_ptr = cyi.data() + 8;

        int ierr = 0;

        int nz;
        ierr = zbessel::zbesj(z.real(), z.imag(), fnu, kode, n, cyr_ptr,
                              cyi_ptr, &nz);

        if (ierr != 0)
          throw std::runtime_error("besj failed with ierr=" +
                                   std::to_string(ierr));

        // Copy results
        for (int j = 0; j < n; ++j) {
          result_ptr[j] = std::complex<double>(cyr_ptr[j], cyi_ptr[j]);
        }

        // check if guards were touched
        for (int j = 0; j < 8; ++j) {
          if (cyr[j] != 0.0 || cyi[j] != 0.0 || cyr[n + 8 + j] != 0.0 ||
              cyi[n + 8 + j] != 0.0) {
            throw std::runtime_error("besh guard zones were modified");
          }
        }

        // Transfer ownership to numpy
        nb::capsule owner(result.release(), [](void *p) noexcept {
          delete[] static_cast<std::complex<double> *>(p);
        });

        size_t shape[1] = {static_cast<size_t>(n)};

        return nb::ndarray<nb::numpy, std::complex<double>>(result_ptr, 1,
                                                            shape, owner);
      },
      nb::arg("z"), nb::arg("fnu"), nb::arg("kode"), nb::arg("n"),
      R"pbdoc(
Compute an n-member sequence of Bessel functions J(fnu + j - 1, z).

Parameters
----------
z : complex
    Complex argument.
fnu : float
    Order of the initial Bessel function (>= 0.0).
kode : int
    Scaling option (1 or 2):
    1 - returns the Bessel function as is
    2 - returns the Bessel function scaled by exp(-abs(z.imag))
n : int
    Number of members in the returned sequence.

Returns
-------
ndarray[complex128]
    1-D numpy array of length n containing J(fnu + j - 1, z) for j=1..n.

Raises
------
RuntimeError
    If the underlying routine reports an error (ierr != 0).
)pbdoc");
}
