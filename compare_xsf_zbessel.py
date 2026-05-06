from pyamos import besh_xsf, besh_zbessel
from pyamos import besj_xsf, besj_zbessel

z = 14.0 - 3.0j
n = 260

result_zbessel = besh_zbessel(z, fnu=1.0, kode=1, m=1, n=n)
result_xsf = besh_xsf(z, fnu=1.0, kode=1, m=1, n=n)

print(result_zbessel)
print(result_xsf)

# doesn't  show the problem, only during "vectorized" calls
# for fnu in range(1, 1 + n + 1):
#     besh_xsf(z, fnu=fnu, kode=1, m=1, n=1)

n = 50
z = 70.26595687262548 - 0.7026595687262548j

result_zbessel = besj_zbessel(z, fnu=0.0, kode=1, n=n)
result_xsf = besj_xsf(z, fnu=0.0, kode=1, n=n)

print(result_zbessel)
print(result_xsf)
