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

    #a=0.074
    #b=4.878
    #a_f=2.628
    #b_f=5.214

    a = 0.074
    b = 4.878
    a_f = 2.628
    b_f = 5.214

    J = ufl.det(F)
    C = pow(J, -float(2) / 3) * F.T * F
    #C = F.T * F
 
    e1 = ufl.as_vector([1.0, 0.0, 0.0])

    IIFx = ufl.tr(C)
    I4e1 = ufl.inner(C * e1, e1)

    cond = lambda a: ufl.conditional(a > 0, a, 0)

    W_hat = a / (2 * b) * (ufl.exp(b * (IIFx - 3)) - 1)
    W_f = a_f / (2 * b_f) * (ufl.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

    return  W_hat + W_f

def psi_neohookean(
    F,
    C10=1e5,
):
    """

    Declares the strain energy function for a neohookean (incompressible) material.

    Args:
        F - deformation tensor

    Returns:
        psi(F), scalar function

    """

    J = ufl.det(F)
    #C = F.T * F
    C = pow(J, -float(2) / 3) * F.T * F

    IIFx = ufl.tr(C)

    return (C10/2)*(IIFx - 3) # add this term for non-isochoric version: - C10*ufl.ln(J)         

def define_weak_form(mesh, stretch_fun):
    """

    Defines function spaces (P1 x P2 x RM) and functions to solve for,
    as well as the weak form for the problem itself.

    Args:
        mesh (df.Mesh): domain to solve equations over
        stretch_fun (ufl form): function that assigns Dirichlet bcs
                on wall to be stretched/extended

    Returns:
        weak form (ufl form), state, displacement, boundary conditions

    """
    
    """
    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

    state_space = df.fem.FunctionSpace(mesh, P2 * P1)
    
    state = df.fem.Function(state_space)
    test_state = ufl.TestFunctions(state_space)

    u, p = state.split()
    v, q = test_state
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
    
    
    lmbda = 1000
    psi = psi_holzapfel(F) + lmbda/2*(J*ufl.ln(J) - J + 1)
    P = ufl.diff(psi, F)
    weak_form = ufl.inner(P, ufl.grad(v)) * dx
    """

    weak_form = 0
    weak_form += elasticity_term(F, J, p, v, dx)
    weak_form += pressure_term(q, J, dx)

    V, _ = state_space.sub(0).collapse()
    """
    
    bcs = define_bcs(V, mesh, stretch_fun)
    state = u
    
    return weak_form, state, bcs


def elasticity_term(F, J, p, v, dx):
    """

    First term of the weak form

    Args:
        F (ufl form): deformation tensor
        J (ufl form): Jacobian
        p (df.Function): pressure
        v (df.TestFunction): test function for displacement

    Returns:
        component of weak form (ufl form)

    """

    psi = psi_holzapfel(F)
    P = ufl.diff(psi, F) + p * J * ufl.inv(F.T)
    
    return ufl.inner(P, ufl.grad(v)) * dx


def pressure_term(q, J, dx):
    """

    Second term of the weak form

    Args:
        q (df.TestFunction): test function for pressure
        J (ufl form): Jacobian

    Returns:
        component of weak form (ufl form)

    """
    return q * (J - 1) * dx


def define_bcs(V, mesh, stretch_fun):
    """

    Defines boundary conditions based on displacement, assuming the domain
    has a box-like shape. We'll keep the displacement on the sides defined
    by lowest x coord, y coords and z coords fixed in their respective
    planes, while stretching the side defined by the highest x coord.

    Args:
        V (df.VectorFunctionSpace): function space for displacement
        mesh (df.Mesh): Domain in which we solve the problem
        stretch_fun (ufl form): function that assigns Dirichlet bcs
                on wall to be stretched/extended

    Returns:
        List of boundary conditions

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

    u_fixed = df.fem.Constant(mesh, PETSc.ScalarType(0))
    
    bnd_funs = [xmin_bnd, ymin_bnd, zmin_bnd]
    components = [0, 1, 2]

    bcs = []

    for bnd_fun, comp in zip(bnd_funs, components):
        boundary_facets = df.mesh.locate_entities_boundary(mesh, fdim, bnd_fun)
        dofs = df.fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        bc = df.fem.dirichletbc(u_fixed, dofs, V.sub(0))
        bcs.append(bc)
        

    # then the moving one

    boundary_facets = df.mesh.locate_entities_boundary(mesh, fdim, xmax_bnd)
    dofs = df.fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc = df.fem.dirichletbc(stretch_fun, dofs, V.sub(0))
    bcs.append(bc)

    return bcs


mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
stretch = np.linspace(0, 0.1, 50)
stretch_fun = df.fem.Constant(mesh, PETSc.ScalarType(0.0))

weak_form, state, bcs = define_weak_form(mesh, stretch_fun)

problem = df.fem.petsc.NonlinearProblem(weak_form, state, bcs)
solver = df.nls.petsc.NewtonSolver(mesh.comm, problem)

solver.rtol=1e-4
solver.atol=1e-4
solver.convergence_criterium = "incremental"


for s in stretch:
    print(f"Domain stretch: {100*s:.5f} %")
    stretch_fun.value = s
    solver.solve(state)
