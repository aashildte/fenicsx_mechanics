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

    Defines function spaces (P1 x P2) and functions to solve for, as well
    as the weak form for the problem itself. This assumes a fully incompressible
    formulation, solving for the displacement and the hydrostatic pressure.

    Args:
        mesh (df.Mesh): domain to solve equations over

    Returns:
        weak form (ufl form), state, displacement, boundary conditions
        active_fun (ufl form): function that imposes active contraction
            through an active strain approach

    """
    
    
    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

    state_space = df.fem.FunctionSpace(mesh, P2 * P1)
    
    state = df.fem.Function(state_space)
    test_state = ufl.TestFunctions(state_space)

    u, p = ufl.split(state)
    v, q = test_state
    
    # Kinematics
    d = len(u)
    I = ufl.Identity(d)                # Identity tensor
    F = ufl.variable(I + ufl.grad(u))  # Deformation gradient
    J = ufl.det(F)

    # active contraction, here given as a constant
    active_fun = df.fem.Constant(mesh, PETSc.ScalarType(0))

    # Weak form
    
    metadata = {"quadrature_degree": 4}
    dx = ufl.Measure("dx", domain=mesh, metadata=metadata)
    
    weak_form = 0
    weak_form += elasticity_term(active_fun, F, J, p, v, dx)
    weak_form += pressure_term(q, J, dx) 
    
    bcs = define_bcs(state_space, mesh)
    
    return weak_form, state, bcs, active_fun


def elasticity_term(active_fun, F, J, p, v, dx):
    """

    First term of the weak form

    Args:
        active_fun (ufl form) : scalar function
        F (ufl form): deformation tensor
        J (ufl form): Jacobian
        p (df.Function): pressure
        v (df.TestFunction): test function for displacement

    Returns:
        component of weak form (ufl form)

    """
    
    sqrt_fun = (1 - active_fun) ** (-0.5)
    F_a = ufl.as_tensor(((1 - active_fun, 0, 0), (0, sqrt_fun, 0), (0, 0, sqrt_fun)))

    F_e = ufl.variable(F * ufl.inv(F_a))
    psi = psi_holzapfel(F_e)

    P = ufl.det(F_a) * ufl.diff(psi, F_e) * ufl.inv(F_a.T) + p * J * ufl.inv(F.T)

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


def define_bcs(state_space, mesh):
    """

    Defines boundary conditions based on displacement, assuming the domain
    has a box-like shape. We'll keep the displacement on the sides defined
    by lowest x coord, y coords and z coords fixed in their respective planes.

    Args:
        state_space (FunctionSpace): function space for displacement and pressure
        mesh (df.Mesh): Domain in which we solve the problem

    Returns:
        List of boundary conditions
    
    """
    

    coords = mesh.geometry.x
    xmin = min(coords[:, 0])
    ymin = min(coords[:, 1])
    zmin = min(coords[:, 2])

    xmin_bnd = lambda x : np.isclose(x[0], xmin)
    ymin_bnd = lambda x : np.isclose(x[1], ymin)
    zmin_bnd = lambda x : np.isclose(x[2], zmin)
    corner = lambda x : np.logical_and(np.logical_and(xmin_bnd(x), ymin_bnd(x)), zmin_bnd(x))

    fdim = 2 
    bcs = []
    
    # fix three of the boundaries in their respective planes
    
    bnd_funs = [xmin_bnd, ymin_bnd, zmin_bnd]
    components = [0, 1, 2]
    
    V0, _ = state_space.sub(0).collapse()    

    for bnd_fun, comp in zip(bnd_funs, components):
        V_c, _ = V0.sub(comp).collapse()
        u_fixed = fem.Function(V_c)
        u_fixed.vector.array[:] = 0
        dofs = fem.locate_dofs_geometrical((state_space.sub(0).sub(comp),V_c), bnd_fun)
        bc = fem.dirichletbc(u_fixed, dofs, state_space.sub(0).sub(comp))
        bcs.append(bc)
    

    # fix corner completely
    
    u_fixed = fem.Function(V0)
    u_fixed.vector.array[:] = 0
    dofs = fem.locate_dofs_geometrical((state_space.sub(0),V0), corner)
    bc = fem.dirichletbc(u_fixed, dofs, state_space.sub(0))
    bcs.append(bc)
    
    return bcs


def active_strain_value(t):
    """

    Define an active strain function as an exponential curve. There
    is no physiological reasoning for using this, but it's simple and
    it looks right; having a steeper upbeat curve followed by a slower
    restoration phase.

    """
    theta = 50
    k = 5
    scaling_parameter = 2
    return scaling_parameter/(theta**k)*t**(k-1)*np.exp(-t/theta)
    

time = np.linspace(0, 200, 200)
active_values = active_strain_value(time)


mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)

weak_form, state, bcs, active_fun = define_weak_form(mesh)

problem = df.fem.petsc.NonlinearProblem(weak_form, state, bcs)
solver = df.nls.petsc.NewtonSolver(mesh.comm, problem)

solver.rtol=1e-4
solver.atol=1e-4
solver.convergence_criterium = "incremental"


fout = df.io.XDMFFile(mesh.comm, "displacement.xdmf", "w")
fout.write_mesh(mesh)


for (i, a) in enumerate(active_values):
    if i%50==0:
        print(f"Active tension value: {a:.2f}; step {i}")
    active_fun.value = a
    solver.solve(state)
    u, _ = state.split()

    fout.write_function(u, a)

fout.close()
