# Import necessary libraries from Firedrake for finite element analysis
# and from icepack for glacier dynamics modeling.
import firedrake
from firedrake import sqrt, exp, max_value, inner, as_vector, Constant, dx, interpolate
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    weertman_sliding_law as m,
)
import icepack


# Define the friction law using the Weertman sliding law.
def friction(**kwargs):
    # Unpack necessary variables from keyword arguments.
    u, h, s, C = map(kwargs.get, ("velocity", "thickness", "surface", "friction"))

    # Calculate effective pressures and basal shear stress.
    p_W = ρ_W * g * max_value(0, -(s - h))  # Water pressure
    p_I = ρ_I * g * h  # Ice overburden pressure
    N = max_value(0, p_I - p_W)  # Effective pressure
    τ_c = N / 2  # Basal shear stress

    # Compute basal sliding velocity and return the friction.
    u_c = (τ_c / C) ** m
    u_b = sqrt(inner(u, u))
    return τ_c * ((u_c ** (1 / m + 1) + u_b ** (1 / m + 1)) ** (m / (m + 1)) - u_c)


# Function to run the simulation over a given time period with specified time step.
def run_simulation(solver, h, s, u, final_time, dt):
    num_steps = int(final_time / dt)  # Calculate the number of time steps
    for step in range(num_steps):
        # Update ice thickness using the prognostic solver.
        h = solver.prognostic_solve(
            dt, thickness=h, velocity=u, accumulation=a, thickness_inflow=h_0
        )
        # Recompute the surface elevation.
        s = icepack.compute_surface(thickness=h, bed=z_b)
        # Solve for ice velocity using the diagnostic solver.
        u = solver.diagnostic_solve(
            velocity=u, thickness=h, surface=s, fluidity=A, friction=C
        )
    return h, s, u  # Return updated thickness, surface, and velocity


default_opts = {
    "dirichlet_ids": [1],
    "side_wall_ids": [3, 4],
    "diagnostic_solver_type": "icepack",
    "diagnostic_solver_parameters": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "tolerance": 1e-8,
    },
}


faster_opts = {
    "dirichlet_ids": [1],
    "side_wall_ids": [3, 4],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "ksp_type": "cg",
        "pc_type": "mg",
        "pc_mg_cycle_type": "w",
        "snes_line_search_type": "cp",
    },
    "prognostic_solver_parameters": {
        "ksp_type": "gmres",
        "pc_type": "ilu",
    },
}

# ----------------------------------------------------------------------------------
# Set dimensions for the computational domain and calculate the number of elements.
Lx, Ly = 640e3, 80e3  # Domain dimensions in meters
ny = 20  # Number of elements along y-axis
nx = int(Lx / Ly) * ny  # Number of elements along x-axis, maintaining aspect ratio

# Create a rectangular mesh for the domain and define function spaces.
mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly)
Q = firedrake.FunctionSpace(mesh, "CG", 1)  # Scalar function space
V = firedrake.VectorFunctionSpace(mesh, "CG", 1)  # Vector function space

# Define spatial coordinates and parameters for bedrock topography.
x, y = firedrake.SpatialCoordinate(mesh)
x_c = Constant(300e3)  # Scaling factor for x-coordinate
X = x / x_c
# Define coefficients for a polynomial bedrock profile in the x-direction.
B_0, B_2, B_4, B_6 = (
    Constant(-150),
    Constant(-728.8),
    Constant(343.91),
    Constant(-50.57),
)
B_x = B_0 + B_2 * X**2 + B_4 * X**4 + B_6 * X**6  # Polynomial bedrock profile

# Parameters for bedrock variation in the y-direction.
f_c, d_c, w_c = Constant(4e3), Constant(500), Constant(24e3)
B_y = d_c * (
    1 / (1 + exp(-2 * (y - Ly / 2 - w_c) / f_c))
    + 1 / (1 + exp(+2 * (y - Ly / 2 + w_c) / f_c))
)

# Interpolate bedrock elevation, ensuring it does not go below a minimum depth.
z_deep = Constant(-720)  # Minimum bedrock depth
z_b = interpolate(max_value(B_x + B_y, z_deep), Q)

# Output the initial bedrock topography to a file.
outfile = firedrake.File("output/bed_rock.pvd")
outfile.write(z_b)

# Constants for the simulation. ----------------------------------------------------
A, C = Constant(20), Constant(1e-2)  # Flow law parameter and sliding coefficient
a = Constant(0.3) # Accumulation rate constant.

# Initialize the ice stream model with the defined friction law.
model = icepack.models.IceStream(friction=friction)

# Interpolate initial thickness and compute the initial surface elevation.
h_0 = interpolate(Constant(100), Q)  # Initial ice thickness
s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)  # Initial surface elevation

# Simulation parameters for initial and subsequent runs.
dt_initial = 3.0  # Initial time step
final_time_initial = 900.0  # Initial simulation time
dt_subsequent = 5.0  # Subsequent time step
final_time_subsequent = 20.0  # Subsequent simulation time


# Create solver instances with specified options.
default_solver = icepack.solvers.FlowSolver(model, **default_opts)
faster_solver = icepack.solvers.FlowSolver(model, **faster_opts)

# Initial run with the default solver.
u_0 = default_solver.diagnostic_solve(
    velocity=interpolate(as_vector((90 * x / Lx, 0)), V),
    thickness=h_0,
    surface=s_0,
    fluidity=A,
    friction=C,
)

h_900, s_900, u_900 = run_simulation(
    default_solver, h_0, s_0, u_0, final_time_initial, dt_initial
)

# # Subsequent runs with fast and faster solvers.
# h, s, u = run_simulation(
#     faster_solver, h_900, s_900, u_900, final_time_subsequent, dt_subsequent
# )

# # Output final results.
# outfile = firedrake.File("output/output.pvd")
# outfile.write(h)
