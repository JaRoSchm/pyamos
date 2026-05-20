#include "amos.h"
#include "zbessel/zbessel.hh"
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <vector>

namespace nb = nanobind;

static void validate_bessel_args(double fnu, int n) {
  if (fnu < 0.0)
    throw std::invalid_argument("fnu must be non-negative");
  if (n <= 0)
    throw std::invalid_argument("n must be positive");
}

template <typename Func>
static nb::ndarray<nb::numpy, std::complex<double>>
bessel_complex_buf(int n, const char *name, Func &&call) {
  auto result = std::make_unique<std::complex<double>[]>(n);
  auto *result_ptr = result.get();

  // add guards
  std::vector<std::complex<double>> cy(n + 16, {0.0, 0.0});
  auto *cy_ptr = cy.data() + 8;

  int ierr = 0;
  // call(result_ptr, &ierr);
  call(cy_ptr, &ierr);

  if (ierr != 0)
    throw std::runtime_error(std::string(name) +
                             " failed with ierr=" + std::to_string(ierr));

  // copy results (for guards)
  for (int j = 0; j < n; ++j)
    result_ptr[j] = cy_ptr[j];

  // check if guards were touched
  for (int j = 0; j < 8; ++j)
    if (cy[j] != std::complex<double>(0, 0) ||
        cy[n + 8 + j] != std::complex<double>(0, 0))
      throw std::runtime_error(std::string(name) +
                               " guard zones were modified");

  // transfer ownership to numpy
  nb::capsule owner(result.release(), [](void *p) noexcept {
    delete[] static_cast<std::complex<double> *>(p);
  });
  size_t shape[1] = {static_cast<size_t>(n)};
  return nb::ndarray<nb::numpy, std::complex<double>>(result_ptr, 1, shape,
                                                      owner);
}

template <typename Func>
static nb::ndarray<nb::numpy, std::complex<double>>
bessel_split_buf(int n, const char *name, Func &&call) {
  auto result = std::make_unique<std::complex<double>[]>(n);
  auto *result_ptr = result.get();

  // add guards
  std::vector<double> cyr(n + 16, 0.0), cyi(n + 16, 0.0);
  auto *cyr_ptr = cyr.data() + 8;
  auto *cyi_ptr = cyi.data() + 8;

  int ierr = 0;
  call(cyr_ptr, cyi_ptr, &ierr);

  if (ierr != 0)
    throw std::runtime_error(std::string(name) +
                             " failed with ierr=" + std::to_string(ierr));

  // copy results (for guards)
  for (int j = 0; j < n; ++j)
    result_ptr[j] = {cyr_ptr[j], cyi_ptr[j]};

  // check if guards were touched
  for (int j = 0; j < 8; ++j)
    if (cyr[j] != 0.0 || cyi[j] != 0.0 || cyr[n + 8 + j] != 0.0 ||
        cyi[n + 8 + j] != 0.0)
      throw std::runtime_error(std::string(name) +
                               " guard zones were modified");

  // transfer ownership to numpy
  nb::capsule owner(result.release(), [](void *p) noexcept {
    delete[] static_cast<std::complex<double> *>(p);
  });
  size_t shape[1] = {static_cast<size_t>(n)};
  return nb::ndarray<nb::numpy, std::complex<double>>(result_ptr, 1, shape,
                                                      owner);
}

NB_MODULE(pyamos, m) {

  // xsf variants

  m.def(
      "besh_xsf",
      [](std::complex<double> z, double fnu, int kode, int m, int n) {
        validate_bessel_args(fnu, n);
        return bessel_complex_buf(n, "besh", [&](auto *cp, int *ierr) {
          int nz = xsf::amos::besh(z, fnu, kode, m, n, cp, ierr);
          if (nz != 0)
            std::cout << "besh_xsf: nz = " << nz << std::endl;
        });
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
        validate_bessel_args(fnu, n);
        return bessel_complex_buf(n, "besj", [&](auto *cp, int *ierr) {
          int nz = xsf::amos::besj(z, fnu, kode, n, cp, ierr);
          if (nz != 0)
            std::cout << "besj_xsf: nz = " << nz << std::endl;
        });
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
      "besy_xsf",
      [](std::complex<double> z, double fnu, int kode, int n) {
        validate_bessel_args(fnu, n);
        return bessel_complex_buf(n, "besy", [&](auto *cp, int *ierr) {
          int nz = xsf::amos::besy(z, fnu, kode, n, cp, ierr);
          if (nz != 0)
            std::cout << "besy_xsf: nz = " << nz << std::endl;
        });
      },
      nb::arg("z"), nb::arg("fnu"), nb::arg("kode"), nb::arg("n"),
      R"pbdoc(FIXME)pbdoc");

  m.def(
      "besk_xsf",
      [](std::complex<double> z, double fnu, int kode, int n) {
        validate_bessel_args(fnu, n);
        return bessel_complex_buf(n, "besk", [&](auto *cp, int *ierr) {
          int nz = xsf::amos::besk(z, fnu, kode, n, cp, ierr);
          if (nz != 0)
            std::cout << "besk_xsf: nz = " << nz << std::endl;
        });
      },
      nb::arg("z"), nb::arg("fnu"), nb::arg("kode"), nb::arg("n"),
      R"pbdoc(FIXME)pbdoc");

  m.def(
      "besi_xsf",
      [](std::complex<double> z, double fnu, int kode, int n) {
        validate_bessel_args(fnu, n);
        return bessel_complex_buf(n, "besi", [&](auto *cp, int *ierr) {
          int nz = xsf::amos::besi(z, fnu, kode, n, cp, ierr);
          if (nz != 0)
            std::cout << "besi_xsf: nz = " << nz << std::endl;
        });
      },
      nb::arg("z"), nb::arg("fnu"), nb::arg("kode"), nb::arg("n"),
      R"pbdoc(FIXME)pbdoc");

  // zbessel variants

  m.def(
      "besh_zbessel",
      [](std::complex<double> z, double fnu, int kode, int m, int n) {
        validate_bessel_args(fnu, n);
        return bessel_split_buf(n, "besh", [&](auto *rr, auto *ri, int *ierr) {
          int nz;
          *ierr =
              zbessel::zbesh(z.real(), z.imag(), fnu, kode, m, n, rr, ri, &nz);
          if (nz != 0)
            std::cout << "besh_zbessel: nz = " << nz << std::endl;
        });
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
      "besj_zbessel",
      [](std::complex<double> z, double fnu, int kode, int n) {
        validate_bessel_args(fnu, n);
        return bessel_split_buf(n, "besj", [&](auto *rr, auto *ri, int *ierr) {
          int nz;
          *ierr = zbessel::zbesj(z.real(), z.imag(), fnu, kode, n, rr, ri, &nz);
          if (nz != 0)
            std::cout << "besj_zbessel: nz = " << nz << std::endl;
        });
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
      "besy_zbessel",
      [](std::complex<double> z, double fnu, int kode, int n) {
        validate_bessel_args(fnu, n);
        return bessel_split_buf(n, "besy", [&](auto *rr, auto *ri, int *ierr) {
          int nz;
          std::vector<double> cwrkr(n), cwrki(n);
          *ierr = zbessel::zbesy(z.real(), z.imag(), fnu, kode, n, rr, ri, &nz,
                                 cwrkr.data(), cwrki.data());
          if (nz != 0)
            std::cout << "besy_zbessel: nz = " << nz << std::endl;
        });
      },
      nb::arg("z"), nb::arg("fnu"), nb::arg("kode"), nb::arg("n"),
      R"pbdoc(FIXME)pbdoc");

  m.def(
      "besk_zbessel",
      [](std::complex<double> z, double fnu, int kode, int n) {
        validate_bessel_args(fnu, n);
        return bessel_split_buf(n, "besk", [&](auto *rr, auto *ri, int *ierr) {
          int nz;
          *ierr = zbessel::zbesk(z.real(), z.imag(), fnu, kode, n, rr, ri, &nz);
          if (nz != 0)
            std::cout << "besk_zbessel: nz = " << nz << std::endl;
        });
      },
      nb::arg("z"), nb::arg("fnu"), nb::arg("kode"), nb::arg("n"),
      R"pbdoc(FIXME)pbdoc");

  m.def(
      "besi_zbessel",
      [](std::complex<double> z, double fnu, int kode, int n) {
        validate_bessel_args(fnu, n);
        return bessel_split_buf(n, "besi", [&](auto *rr, auto *ri, int *ierr) {
          int nz;
          *ierr = zbessel::zbesi(z.real(), z.imag(), fnu, kode, n, rr, ri, &nz);
          if (nz != 0)
            std::cout << "besi_zbessel: nz = " << nz << std::endl;
        });
      },
      nb::arg("z"), nb::arg("fnu"), nb::arg("kode"), nb::arg("n"),
      R"pbdoc(FIXME)pbdoc");
}

