import numpy as np
from pyamos import besh_xsf, besh_zbessel
from pyamos import besj_xsf, besj_zbessel

z = 14.0 - 3.0j
n = 260

result_zbessel = besh_zbessel(z, fnu=1.0, kode=1, m=1, n=n)
print("------")
result_xsf = besh_xsf(z, fnu=1.0, kode=1, m=1, n=n)

np.testing.assert_allclose(result_xsf, result_zbessel, atol=0, rtol=1e-13)

# doesn't  show the problem, only during "vectorized" calls
# for fnu in range(1, 1 + n + 1):
#     besh_xsf(z, fnu=fnu, kode=1, m=1, n=1)

n = 50
z = 70.26595687262548 - 0.7026595687262548j

result_zbessel = besj_zbessel(z, fnu=0.0, kode=1, n=n)
print("------")
result_xsf = besj_xsf(z, fnu=0.0, kode=1, n=n)

np.testing.assert_allclose(result_xsf, result_zbessel, atol=0, rtol=1e-13)

n = 1000
z = -50.0 - 50.0j

result_zbessel = besj_zbessel(z, fnu=0.0, kode=1, n=n)
print("------")
result_xsf = besj_xsf(z, fnu=0.0, kode=1, n=n)

np.testing.assert_allclose(result_xsf, result_zbessel, atol=0, rtol=1e-13)
