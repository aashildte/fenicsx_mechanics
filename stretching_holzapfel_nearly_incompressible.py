"""

Åshild Telle / Simula Research Laboratory / 2022

"""

import numpy as np
import dolfinx as df
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, plot, nls, cpp, log


def psi_holzapfel(
    F,
):
    """

    Declares the strain energy function for a simplified holzapfel formulation.

    Args:
        F - deformation tensor

    Returns:
        psi(F), scalar function

    """

    a = 0.074
    b = 4.878
    a_f = 2.628
    b_f = 5.214

    J = ufl.det(F)
    C = pow(J, -float(2) / 3) * F.T * F
 
    e1 = ufl.as_vector([1.0, 0.0, 0.0])

    IIFx = ufl.tr(C)
    I4e1 = ufl.inner(C * e1, e1)

    cond = lambda a: ufl.conditional(a > 0, a, 0)

    W_hat = a / (2 * b) * (ufl.exp(b * (IIFx - 3)) - 1)
    W_f = a_f / (2 * b_f) * (ufl.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

    return  W_hat + W_f


def define_weak_form(mesh):
    """

    Defines function spaces, functions to solve for, as well as the weak form for
    the problem itself. This assumes a nearly incompressible formulation.

    Args:
        mesh (df.Mesh): domain to solve equations over

    Returns:
        weak form (ufl form), state, displacement, boundary conditions
        stretch_fun (ufl form): function that assigns Dirichlet bcs
                on wall to be stretched/extended

    """
    
    V = df.fem.VectorFunctionSpace(mesh, ("CG", 2))
    v = ufl.TestFunction(V)
    u = df.fem.Function(V)
    
    # Kinematics
    d = len(u)
    I = ufl.Identity(d)                # Identity tensor
    F = ufl.variable(I + ufl.grad(u))  # Deformation gradient
    J = ufl.det(F)
    
    # Weak form
    
    metadata = {"quadrature_degree": 4}
    dx = ufl.Measure("dx", domain=mesh, metadata=metadata)
    
    kappa = 1000
    psi = psi_holzapfel(F) + kappa/2*(J*ufl.ln(J) - J + 1)
    P = ufl.diff(psi, F)
    weak_form = ufl.inner(P, ufl.grad(v)) * dx
    
    bcs, stretch_fun = define_bcs(V, mesh)
    
    return weak_form, u, bcs, stretch_fun


def define_bcs(V, mesh):
    """
    Defines boundary conditions based on displacement, assuming the domain
    has a box-like shape. We'll keep the displacement on the sides defined
    by lowest x coord, y coords and z coords fixed in their respective
    planes, while stretching the side defined by the highest x coord.

    Args:
        V (df.VectorFunctionSpace): function space for displacement
        mesh (df.Mesh): Domain in which we solve the problem
    
    Returns:
        List of boundary conditions
        stretch_fun (ufl form): function that assigns Dirichlet bcs
                on wall to be stretched/extended
    
    """

    coords = mesh.geometry.x
    xmin = min(coords[:, 0])
    xmax = max(coords[:, 0])
    ymin = min(coords[:, 1])
    zmin = min(coords[:, 2])

    xmin_bnd = lambda x : np.isclose(x[0], xmin)
    xmax_bnd = lambda x : np.isclose(x[0], xmax)
    ymin_bnd = lambda x : np.isclose(x[1], ymin)
    zmin_bnd = lambda x : np.isclose(x[2], zmin)

    fdim = 2

    # first define the fixed boundaries

    
    bnd_funs = [xmin_bnd, ymin_bnd, zmin_bnd]
    components = [0, 1, 2]

    bcs = []

    for bnd_fun, comp in zip(bnd_funs, components):
        V0, _ = V.sub(comp).collapse()
    
        fixed_fun = fem.Function(V0)
        fixed_fun.vector.array[:] = 0

        boundary_facets = df.mesh.locate_entities_boundary(mesh, fdim, bnd_fun)
        dofs = df.fem.locate_dofs_topological((V.sub(comp), V0), fdim, boundary_facets)
        
        bc = df.fem.dirichletbc(fixed_fun, dofs, V.sub(comp))
        bcs.append(bc)

    # then the moving one
    V0, _ = V.sub(0).collapse()
    stretch_fun = fem.Function(V0)
    stretch_fun.vector.array[:] = 0

    boundary_facets = df.mesh.locate_entities_boundary(mesh, fdim, xmax_bnd)
    dofs = df.fem.locate_dofs_topological((V.sub(0), V0), fdim, boundary_facets)
    
    bc = df.fem.dirichletbc(stretch_fun, dofs, V.sub(0))
    bcs.append(bc)

    return bcs, stretch_fun


mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)

weak_form, u, bcs, stretch_fun = define_weak_form(mesh)

problem = df.fem.petsc.NonlinearProblem(weak_form, u, bcs)
solver = df.nls.petsc.NewtonSolver(mesh.comm, problem)

solver.rtol=1e-4
solver.atol=1e-4
solver.convergence_criterium = "incremental"

stretch_values = np.linspace(0, 0.2, 10)

fout = df.io.XDMFFile(mesh.comm, "displacement.xdmf", "w")
fout.write_mesh(mesh)

for s in stretch_values:
    print(f"Domain stretch: {100*s:.5f} %")
    stretch_fun.vector.array[:] = s
    solver.solve(u)

    fout.write_function(u, s)

fout.close()
