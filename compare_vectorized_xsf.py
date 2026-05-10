import itertools

import numpy as np
from pyamos import besh_xsf, besj_xsf

NU_VALUES = [0, 0.25, 0.5]

N_VALUES = [1000]

UNSCALED_Z = [
    -50,
    -40,
    -30,
    -25,
    -20,
    -15,
    -12,
    -10,
    -8,
    -6,
    -4,
    -3,
    -2,
    -1,
    -0.5,
    -0.1,
    -0.001,
    -1e-6,
    0,
    1e-6,
    0.001,
    0.1,
    0.5,
    1,
    2,
    3,
    4,
    6,
    8,
    10,
    12,
    15,
    20,
    25,
    30,
    40,
    50,
]

SCALED_Z = [
    -1000,
    -500,
    -300,
    -200,
    -100,
    -50,
    -30,
    -10,
    -5,
    -1,
    -0.1,
    -0.01,
    0.001,
    0.01,
    0.1,
    1,
    5,
    10,
    30,
    50,
    100,
    200,
    300,
    500,
    1000,
]

FUNCTIONS = {
    "J_nu": lambda z, nu, n: besj_xsf(z, nu, 1, n),
    # no problem:
    # "H^1_nu": lambda z, nu, n: besh_xsf(z, nu, 1, 1, n),
    # "H^2_nu": lambda z, nu, n: besh_xsf(z, nu, 1, 2, n),
}

FUNCTIONS_SCALED = {
    "J_nu (scaled)": lambda z, nu, n: besj_xsf(z, nu, 1, n),
    "H^1_nu (scaled)": lambda z, nu, n: besh_xsf(z, nu, 1, 1, n),
    "H^2_nu (scaled)": lambda z, nu, n: besh_xsf(z, nu, 1, 2, n),
}


def do_comparison(func_name, func, z, nu, n, rtol):
    print(f"{func_name = }, {z = }, {nu = }, {n = }")

    error = False

    try:
        result_vec = func(z, nu, n)
    except RuntimeError:
        error = True

    result = np.zeros(n, dtype=complex)

    for i in range(n):
        try:
            result[i] = func(z, nu + i, 1)[0]
        except RuntimeError:
            if error:
                break

            raise RuntimeError(
                "Error occurred for scalar but not for vectorized version."
            )
    else:
        np.testing.assert_allclose(result_vec, result, rtol=rtol, atol=0)


def main():
    for func_name, func, z_real, z_imag, nu, n in itertools.product(
        FUNCTIONS.keys(),
        FUNCTIONS.values(),
        UNSCALED_Z,
        UNSCALED_Z,
        NU_VALUES,
        N_VALUES,
    ):
        z = z_real + 1j * z_imag
        do_comparison(func_name, func, z, nu, n, rtol=1e-10)

    for func_name, func, z_real, z_imag, nu, n in itertools.product(
        FUNCTIONS_SCALED.keys(),
        FUNCTIONS_SCALED.values(),
        SCALED_Z,
        SCALED_Z,
        NU_VALUES,
        N_VALUES,
    ):
        z = z_real + 1j * z_imag
        do_comparison(func_name, func, z, nu, n, rtol=1e-10)


if __name__ == "__main__":
    main()
