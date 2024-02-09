from firedrake import *
from firedrake.petsc import PETSc

mesh = UnitSquareMesh(1000, 1000)
PETSc.Sys.Print('  rank %d owns %d elements and can access %d vertices' \
                % (mesh.comm.rank, mesh.num_cells(), mesh.num_vertices()),
                comm=COMM_SELF)


V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x,y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (dot(grad(v), grad(u)) + v * u) * dx
L = f * v * dx

u = Function(V)
solve(a == L, u, 
      solver_parameters={
          'ksp_type': 'cg',  # Use CG solver
          'pc_type': 'hypre',  # Use algebraic multigrid preconditioning
      })
