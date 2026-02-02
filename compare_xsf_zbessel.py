import numpy as np
from pyamos import besh_xsf, besh_zbessel

z = 14.0 - 3.0j
n = 260

result_zbessel = besh_zbessel(z, fnu=1.0, kode=1, m=1, n=n)
result_xsf = besh_xsf(z, fnu=1.0, kode=1, m=1, n=n)

# doesn't does show the problem, only during "vectorized" calls
# for fnu in range(1, 1 + n + 1):
#     besh_xsf(z, fnu=fnu, kode=1, m=1, n=1)
