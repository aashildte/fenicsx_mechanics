
import dolfinx as df
from mpi4py import MPI
from petsc4py import PETSc
import ufl

mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
constant = df.fem.Constant(mesh, PETSc.ScalarType(1))
form = df.fem.form(constant*ufl.dx(mesh))

volume = df.fem.assemble_scalar(form)

print(f"The volume of the domain is {volume} [unit]^3.")
