import numpy as np
from pyamos import (
    besh_xsf,
    besh_zbessel,
    besj_xsf,
    besj_zbessel,
    besy_xsf,
    besy_zbessel,
    besk_xsf,
    besk_zbessel,
    besi_xsf,
    besi_zbessel,
)

# fixed by xsf#92
z = 14.0 - 3.0j
n = 260
n = 10

result_zbessel = besh_zbessel(z, fnu=250.0, kode=1, m=1, n=n)
print("------")
result_xsf = besh_xsf(z, fnu=1.0, kode=1, m=1, n=n)

np.testing.assert_allclose(result_xsf, result_zbessel, atol=0, rtol=1e-12)

# fixed by xsf#138
n = 15
z = 70.0 - 0.7j

result_zbessel = besj_zbessel(z, fnu=0.0, kode=1, n=n)
print("------")
result_xsf = besj_xsf(z, fnu=0.0, kode=1, n=n)

np.testing.assert_allclose(result_xsf, result_zbessel, atol=0, rtol=1e-13)

# fixed by xsf#145
n = 1000
z = -50.0 - 50.0j

result_zbessel = besj_zbessel(z, fnu=0.0, kode=1, n=n)
print("------")
result_xsf = besj_xsf(z, fnu=0.0, kode=1, n=n)

np.testing.assert_allclose(result_xsf, result_zbessel, atol=0, rtol=1e-13)

# TODO: PR to xsf (x >= 0 in line 1776)
z = 0 - 15j
fnu = 0.0
kode = 1
n = 1000

result_zbessel = besi_zbessel(z, fnu, kode, n)
print("------")
result_xsf = besi_xsf(z, fnu, kode, n)

np.testing.assert_allclose(result_xsf, result_zbessel, atol=0, rtol=1e-13)
