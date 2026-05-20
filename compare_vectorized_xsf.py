import itertools

import numpy as np
from pyamos import besh_xsf, besj_xsf, besy_xsf, besk_xsf, besi_xsf

NU_VALUES = [0.0, 0.25, 0.5]

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
    # "J_nu": lambda z, nu, n: besj_xsf(z, nu, 1, n),
    # "H^1_nu": lambda z, nu, n: besh_xsf(z, nu, 1, 1, n),
    # "H^2_nu": lambda z, nu, n: besh_xsf(z, nu, 1, 2, n),
    # "Y_nu": lambda z, nu, n: besy_xsf(z, nu, 1, n),
    # "K_nu": lambda z, nu, n: besk_xsf(z, nu, 1, n),
    # "I_nu": lambda z, nu, n: besi_xsf(z, nu, 1, n),
}

FUNCTIONS_SCALED = {
    # "J_nu (scaled)": lambda z, nu, n: besj_xsf(z, nu, 1, n),
    # "H^1_nu (scaled)": lambda z, nu, n: besh_xsf(z, nu, 1, 1, n),
    # "H^2_nu (scaled)": lambda z, nu, n: besh_xsf(z, nu, 1, 2, n),
    # "Y_nu (scaled)": lambda z, nu, n: besy_xsf(z, nu, 2, n),
    # "K_nu (scaled)": lambda z, nu, n: besk_xsf(z, nu, 2, n),
    # TODO:
    # "I_nu (scaled)": lambda z, nu, n: besi_xsf(z, nu, 2, n),
}

SKIP = [
    # one element more is set to 0, but same in zbessel
    ("J_nu", -15 - 50j, 0.0, 1000),
    ("J_nu", -15 + 50j, 0.0, 1000),
    ("J_nu", 15 - 50j, 0.0, 1000),
    ("J_nu", 15 + 50j, 0.0, 1000),
    ("J_nu", -8 - 15j, 0.5, 1000),
    ("J_nu", -8 + 15j, 0.5, 1000),
    ("J_nu", 8 - 15j, 0.5, 1000),
    ("J_nu", 8 + 15j, 0.5, 1000),
    ("J_nu", -1e-6 - 1e-6j, 0.0, 1000),
    ("J_nu", -1e-6 + 1e-6j, 0.0, 1000),
    ("J_nu", 1e-6 - 1e-6j, 0.0, 1000),
    ("J_nu", 1e-6 + 1e-6j, 0.0, 1000),
    ("J_nu (scaled)", -0.01 - 0.01j, 0.0, 1000),
    ("J_nu (scaled)", -0.01 + 0.01j, 0.0, 1000),
    ("J_nu (scaled)", 0.01 - 0.01j, 0.0, 1000),
    ("J_nu (scaled)", 0.01 + 0.01j, 0.0, 1000),
    ("J_nu (scaled)", -500 + 1000j, 0.0, 1000),
    ("J_nu (scaled)", -500 + 1000j, 0.25, 1000),
    ("J_nu (scaled)", -500 + 1000j, 0.5, 1000),
    ("J_nu (scaled)", -500 - 1000j, 0.0, 1000),
    ("J_nu (scaled)", -500 - 1000j, 0.25, 1000),
    ("J_nu (scaled)", -500 - 1000j, 0.5, 1000),
    ("J_nu (scaled)", 500 + 1000j, 0.0, 1000),
    ("J_nu (scaled)", 500 + 1000j, 0.25, 1000),
    ("J_nu (scaled)", 500 + 1000j, 0.5, 1000),
    ("J_nu (scaled)", 500 - 1000j, 0.0, 1000),
    ("J_nu (scaled)", 500 - 1000j, 0.25, 1000),
    ("J_nu (scaled)", 500 - 1000j, 0.5, 1000),
    ("J_nu (scaled)", -200 + 1000j, 0.0, 1000),
    ("J_nu (scaled)", -200 + 1000j, 0.25, 1000),
    ("J_nu (scaled)", -200 + 1000j, 0.5, 1000),
    ("J_nu (scaled)", -200 - 1000j, 0.0, 1000),
    ("J_nu (scaled)", -200 - 1000j, 0.25, 1000),
    ("J_nu (scaled)", -200 - 1000j, 0.5, 1000),
    ("J_nu (scaled)", 200 + 1000j, 0.0, 1000),
    ("J_nu (scaled)", 200 + 1000j, 0.25, 1000),
    ("J_nu (scaled)", 200 + 1000j, 0.5, 1000),
    ("J_nu (scaled)", 200 - 1000j, 0.0, 1000),
    ("J_nu (scaled)", 200 - 1000j, 0.25, 1000),
    ("J_nu (scaled)", 200 - 1000j, 0.5, 1000),
    ("H^1_nu (scaled)", -500 + 1000j, 0.0, 1000),
    ("H^1_nu (scaled)", -500 + 1000j, 0.25, 1000),
    ("H^1_nu (scaled)", -500 + 1000j, 0.5, 1000),
    ("H^1_nu (scaled)", -500 - 1000j, 0.0, 1000),
    ("H^1_nu (scaled)", -500 - 1000j, 0.25, 1000),
    ("H^1_nu (scaled)", -500 - 1000j, 0.5, 1000),
    ("H^1_nu (scaled)", 500 + 1000j, 0.0, 1000),
    ("H^1_nu (scaled)", 500 + 1000j, 0.25, 1000),
    ("H^1_nu (scaled)", 500 + 1000j, 0.5, 1000),
    ("H^1_nu (scaled)", 500 - 1000j, 0.0, 1000),
    ("H^1_nu (scaled)", 500 - 1000j, 0.25, 1000),
    ("H^1_nu (scaled)", 500 - 1000j, 0.5, 1000),
    ("H^1_nu (scaled)", -200 + 1000j, 0.0, 1000),
    ("H^1_nu (scaled)", -200 + 1000j, 0.25, 1000),
    ("H^1_nu (scaled)", -200 + 1000j, 0.5, 1000),
    ("H^1_nu (scaled)", -200 - 1000j, 0.0, 1000),
    ("H^1_nu (scaled)", -200 - 1000j, 0.25, 1000),
    ("H^1_nu (scaled)", -200 - 1000j, 0.5, 1000),
    ("H^1_nu (scaled)", 200 + 1000j, 0.0, 1000),
    ("H^1_nu (scaled)", 200 + 1000j, 0.25, 1000),
    ("H^1_nu (scaled)", 200 + 1000j, 0.5, 1000),
    ("H^1_nu (scaled)", 200 - 1000j, 0.0, 1000),
    ("H^1_nu (scaled)", 200 - 1000j, 0.25, 1000),
    ("H^1_nu (scaled)", 200 - 1000j, 0.5, 1000),
    ("H^2_nu (scaled)", -500 + 1000j, 0.0, 1000),
    ("H^2_nu (scaled)", -500 + 1000j, 0.25, 1000),
    ("H^2_nu (scaled)", -500 + 1000j, 0.5, 1000),
    ("H^2_nu (scaled)", -500 - 1000j, 0.0, 1000),
    ("H^2_nu (scaled)", -500 - 1000j, 0.25, 1000),
    ("H^2_nu (scaled)", -500 - 1000j, 0.5, 1000),
    ("H^2_nu (scaled)", 500 + 1000j, 0.0, 1000),
    ("H^2_nu (scaled)", 500 + 1000j, 0.25, 1000),
    ("H^2_nu (scaled)", 500 + 1000j, 0.5, 1000),
    ("H^2_nu (scaled)", 500 - 1000j, 0.0, 1000),
    ("H^2_nu (scaled)", 500 - 1000j, 0.25, 1000),
    ("H^2_nu (scaled)", 500 - 1000j, 0.5, 1000),
    ("H^2_nu (scaled)", -200 + 1000j, 0.0, 1000),
    ("H^2_nu (scaled)", -200 + 1000j, 0.25, 1000),
    ("H^2_nu (scaled)", -200 + 1000j, 0.5, 1000),
    ("H^2_nu (scaled)", -200 - 1000j, 0.0, 1000),
    ("H^2_nu (scaled)", -200 - 1000j, 0.25, 1000),
    ("H^2_nu (scaled)", -200 - 1000j, 0.5, 1000),
    ("H^2_nu (scaled)", 200 + 1000j, 0.0, 1000),
    ("H^2_nu (scaled)", 200 + 1000j, 0.25, 1000),
    ("H^2_nu (scaled)", 200 + 1000j, 0.5, 1000),
    ("H^2_nu (scaled)", 200 - 1000j, 0.0, 1000),
    ("H^2_nu (scaled)", 200 - 1000j, 0.25, 1000),
    ("H^2_nu (scaled)", 200 - 1000j, 0.5, 1000),
    ("I_nu (scaled)", -500 - 200j, 0.0, 1000),
    ("I_nu (scaled)", -500 - 200j, 0.25, 1000),
    ("I_nu (scaled)", -500 - 200j, 0.5, 1000),
    ("I_nu (scaled)", -500 - 100j, 0.0, 1000),
    ("I_nu (scaled)", -500 - 100j, 0.25, 1000),
    ("I_nu (scaled)", -500 - 100j, 0.5, 1000),
    ("I_nu (scaled)", -500 - 50j, 0.0, 1000),
    ("I_nu (scaled)", -500 - 50j, 0.25, 1000),
    ("I_nu (scaled)", -500 - 50j, 0.5, 1000),
    ("I_nu (scaled)", -500 - 10j, 0.0, 1000),
    ("I_nu (scaled)", -500 - 10j, 0.25, 1000),
    ("I_nu (scaled)", -500 - 10j, 0.5, 1000),
    ("I_nu (scaled)", -500 - 0.1j, 0.0, 1000),
    ("I_nu (scaled)", -500 - 0.1j, 0.25, 1000),
    ("I_nu (scaled)", -500 - 0.1j, 0.5, 1000),
    ("I_nu (scaled)", -500 - 0.01j, 0.0, 1000),
    ("I_nu (scaled)", -500 - 0.01j, 0.25, 1000),
    ("I_nu (scaled)", -500 - 0.01j, 0.5, 1000),
    ("I_nu (scaled)", -500 + 200j, 0.0, 1000),
    ("I_nu (scaled)", -500 + 200j, 0.25, 1000),
    ("I_nu (scaled)", -500 + 200j, 0.5, 1000),
    ("I_nu (scaled)", -500 + 100j, 0.0, 1000),
    ("I_nu (scaled)", -500 + 100j, 0.25, 1000),
    ("I_nu (scaled)", -500 + 100j, 0.5, 1000),
    ("I_nu (scaled)", -500 + 50j, 0.0, 1000),
    ("I_nu (scaled)", -500 + 50j, 0.25, 1000),
    ("I_nu (scaled)", -500 + 50j, 0.5, 1000),
    ("I_nu (scaled)", -500 + 10j, 0.0, 1000),
    ("I_nu (scaled)", -500 + 10j, 0.25, 1000),
    ("I_nu (scaled)", -500 + 10j, 0.5, 1000),
    ("I_nu (scaled)", -500 + 0.1j, 0.0, 1000),
    ("I_nu (scaled)", -500 + 0.1j, 0.25, 1000),
    ("I_nu (scaled)", -500 + 0.1j, 0.5, 1000),
    ("I_nu (scaled)", -500 + 0.01j, 0.0, 1000),
    ("I_nu (scaled)", -500 + 0.01j, 0.25, 1000),
    ("I_nu (scaled)", -500 + 0.01j, 0.5, 1000),
    ("I_nu (scaled)", -500 + 0.001j, 0.0, 1000),
    ("I_nu (scaled)", -500 + 0.001j, 0.25, 1000),
    ("I_nu (scaled)", -500 + 0.001j, 0.5, 1000),
    # ierr=2
    ("H^1_nu", -15 - 50j, 0.0, 1000),
    # rel. error compared to zbessel 4e-10
    ("J_nu (scaled)", -1000 + 0.001j, 0.0, 1000),
    ("J_nu (scaled)", 1000 + 0.001j, 0.0, 1000),
    ("J_nu (scaled)", -1000 + 0.001j, 0.5, 1000),
    ("J_nu (scaled)", 1000 + 0.001j, 0.5, 1000),
    ("J_nu (scaled)", -500 + 0.001j, 0.5, 1000),
    ("J_nu (scaled)", 500 + 0.001j, 0.5, 1000),
    ("J_nu (scaled)", -300 + 0.001j, 0.25, 1000),
    ("J_nu (scaled)", 300 + 0.001j, 0.25, 1000),
    ("H^1_nu (scaled)", -1000 + 0.001j, 0.0, 1000),
    ("Y_nu (scaled)", -1000 + 0.001j, 0.5, 1000),
    ("Y_nu (scaled)", 1000 + 0.001j, 0.0, 1000),
    ("Y_nu (scaled)", 1000 + 0.001j, 0.25, 1000),
    ("Y_nu (scaled)", 1000 + 0.001j, 0.5, 1000),
    ("Y_nu (scaled)", -1000 - 0.01j, 0.0, 1000),
    ("Y_nu (scaled)", 1000 - 0.01j, 0.0, 1000),
    ("Y_nu (scaled)", 1000 + 0.01j, 0.0, 1000),
    ("Y_nu (scaled)", -500 - 0.01j, 0.5, 1000),
    ("Y_nu (scaled)", -500 + 0.01j, 0.5, 1000),
    ("Y_nu (scaled)", -500 + 0.001j, 0.5, 1000),
    ("Y_nu (scaled)", 500 - 0.01j, 0.25, 1000),
    ("Y_nu (scaled)", 500 - 0.01j, 0.5, 1000),
    ("Y_nu (scaled)", 500 + 0.01j, 0.25, 1000),
    ("Y_nu (scaled)", 500 + 0.01j, 0.5, 1000),
    ("Y_nu (scaled)", 500 + 0.001j, 0.25, 1000),
    ("Y_nu (scaled)", 500 + 0.001j, 0.5, 1000),
    ("I_nu", -50 - 15j, 0.0, 1000),
    ("I_nu", -50 + 15j, 0.0, 1000),
    ("I_nu", 50 - 15j, 0.0, 1000),
    ("I_nu", 50 + 15j, 0.0, 1000),
    ("I_nu", -15 - 8j, 0.5, 1000),
    ("I_nu", -15 + 8j, 0.5, 1000),
    ("I_nu", 15 - 8j, 0.5, 1000),
    ("I_nu", 15 + 8j, 0.5, 1000),
    ("I_nu", -1e-6 - 1e-6j, 0.0, 1000),
    ("I_nu", -1e-6 + 1e-6j, 0.0, 1000),
    ("I_nu", 1e-6 - 1e-6j, 0.0, 1000),
    ("I_nu", 1e-6 + 1e-6j, 0.0, 1000),
]


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
        if (func_name, z, nu, n) in SKIP:
            continue
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
        if (func_name, z, nu, n) in SKIP:
            continue
        do_comparison(func_name, func, z, nu, n, rtol=1e-10)


if __name__ == "__main__":
    main()
