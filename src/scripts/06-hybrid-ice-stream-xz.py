#!/usr/bin/env python
# coding: utf-8

# # Ice streams, once more
# 
# In this demo, we'll revisit the ice stream scenario from before but with a different model.
# The model that we'll use here can resolve both *plug flow* and *shear flow*.
# The velocity is a plug flow if the velocity is roughly constanty with depth, whereas in shear flow the speed at the ice base is much smaller than at the ice surface.
# Deep in the interior of an ice sheet, most flow is by vertical shear, whereas the fast ice streams near the margins are roughly in plug flow.
# The shallow stream equations, and the `IceStream` class we used to solve them in the third demo, assume plug flow and thus are limited in the regions where they can be applied without substantial error.
# The `HybridModel` class we'll use here has no such limitation.

# ### Geometry and input data
# 
# We'll use the same footprint geometry as the ice stream demo, but since we're working with a flowband instead of a flowline model, we'll need a mesh that resolves the $z$ direction as well as $x$.
# The model we will use is posed entirely in *terrain-following coordinates*, where the $z$-axis is remapped into the interval $[0, 1]$.
# This new coordinate is defined as
# 
# $$\zeta = \frac{z - b}{h}$$
# 
# and the inverse transformation is $z = h\zeta + b$.
# In the absence of terminus advance or iceberg calving, the transformation to this coordinate system makes the logical geometry of problem completely steady even though the surface can be rising or falling.
# 
# Firedrake includes built-in support generating *extruded meshes* out of a footprint mesh.
# Rather than use a large number of vertical layers to achieve high resolution of functions defined on this mesh, we'll use only one layer but high-degree basis functions.

# In[ ]:


import firedrake

Lx = 50e3
nx = 48
mesh1d = firedrake.IntervalMesh(nx, Lx)
mesh = firedrake.ExtrudedMesh(mesh1d, layers=1)


# To define a function space on an extruded mesh, the only change over 1D function spaces is the addition of the `vfamily` and `vdegree` arguments.
# These describe respectively the finite element type and polynomial degree in the vertical direction.
# The surface and bed elevation only have to be defined on the footprint mesh, so we'll use vertical basis functions that are constant.

# In[ ]:


Q = firedrake.FunctionSpace(
    mesh, "CG", 2, vfamily="R", vdegree=0
)


# The function space for the velocities is more interesting.
# Rather than use the usual continuous or discontinuous Galerkin finite element bases in the vertical, we'll use *Gauss-Legendre* elements.
# This finite element uses [Legendre polynomials](https://en.wikipedia.org/wiki/Legendre_polynomials) instead of the Lagrange interpolating polynomials.
# The Legendre polynomials are advantageous because they are mutually orthogonal.

# In[ ]:


V = firedrake.FunctionSpace(
    mesh, "CG", 2, vfamily="GL", vdegree=2
)


# Creating the input data is mostly the same as before.
# The `SpatialCoordinate` function now returns an additional symbol representing the terrain-following coordinate $\zeta$.

# In[ ]:


from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    weertman_sliding_law as m,
    gravity as g,
)

x, ζ = firedrake.SpatialCoordinate(mesh)


# In[ ]:


b_in, b_out = 200, -400
b = firedrake.interpolate(b_in - (b_in - b_out) * x / Lx, Q)

s_in, s_out = 850, 50
s0 = firedrake.interpolate(s_in - (s_in - s_out) * x / Lx, Q)

h0 = firedrake.interpolate(s0 - b, Q)


# In[ ]:


h_in = s_in - b_in
δs_δx = (s_out - s_in) / Lx
τ_D = -ρ_I * g * h_in * δs_δx


# In[ ]:


u_in, u_out = 20, 2400
velocity_x = u_in + (u_out - u_in) * (x/Lx)**2
u0 = firedrake.interpolate(velocity_x, V)


# In[ ]:


import icepack

T = firedrake.Constant(255.0)
A = icepack.rate_factor(T)


# In[ ]:


C = firedrake.interpolate((0.95 - 0.05 * x/Lx) * τ_D / u_in**(1/m), Q)

p_W = ρ_W * g * firedrake.max_value(0, h0 - s0)
p_I = ρ_I * g * h0
ϕ = 1 - p_W / p_I


# The bed friction can be re-parameterized for the hybrid model in the same way as for the ice stream model.

# In[ ]:


import icepack.models.hybrid

def friction(**kwargs):
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    s = kwargs["surface"]
    C = kwargs["friction"]

    p_W = ρ_W * g * firedrake.max_value(0, h - s)
    p_I = ρ_I * g * h
    ϕ = 1 - p_W / p_I
    return icepack.models.hybrid.bed_friction(
        velocity=u,
        friction=C * ϕ,
    )


# The setup is the same as in the ice stream case, only here we're using `HybridModel` instead of `IceStream`.
# Creating a `FlowSolver` works exactly the same as it did before no matter what model we're using.

# In[ ]:


model = icepack.models.HybridModel(friction=friction)
opts = {
    "dirichlet_ids": [1],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {"snes_type": "newtontr"},
}
solver = icepack.solvers.FlowSolver(model, **opts)

u0 = solver.diagnostic_solve(
    velocity=u0,
    thickness=h0,
    surface=s0,
    fluidity=A,
    friction=C,
)


# One of the major differences between working with flowline and flowband models is that the fields are posed on different geometries and so the kinds of plots we can make are different.
# Firedrake can make contour plots of flowband fields, but often we only care about changes in the horizontal dimension.
# The function `depth_average` will integrate a flowband field with respect to $\zeta$ and give us back the corresponding 1D function, which we can then make line plots of.
# (At the end of this demo we'll show how to make plots of the vertical variability.)

# In[ ]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots()
axes.set_xlabel("distance (m)")
axes.set_ylabel("speed (m / yr)")
firedrake.plot(icepack.depth_average(u0), axes=axes);


# ### Modeling
# 
# The code to run the hybrid model is identical to that of the ice stream demo.

# In[ ]:


import tqdm

num_years = 250
timesteps_per_year = 2

δt = 1.0 / timesteps_per_year
num_timesteps = num_years * timesteps_per_year

a = firedrake.interpolate(1.7 - 2.7 * x / Lx, Q)
h = h0.copy(deepcopy=True)
u = u0.copy(deepcopy=True)

for step in tqdm.trange(num_timesteps):
    h = solver.prognostic_solve(
        δt,
        thickness=h,
        velocity=u,
        accumulation=a,
        thickness_inflow=h0,
    )
    s = icepack.compute_surface(thickness=h, bed=b)
    
    u = solver.diagnostic_solve(
        velocity=u,
        thickness=h,
        surface=s,
        fluidity=A,
        friction=C,
    )


# As a sanity check, we can plot the thickness, velocity, and mass balance.
# These give very similar results in the eyeball norm to our previous demo assuming only plug flow.
# Note again that there are some wiggles on the upstream boundary.
# This is the expected rapid adjustment of the ice within a few kilometers of the upstream boundary when the system isn't quite in equilibrium with the inflow values.
# In a later notebook, we'll show how to relax the inflow values to equilibrate the two.
# 
# We're calling the `depth_average` function on the ice thickness below, which might seem a little weird.
# Again, this is just a trick to get the thickness field, which is strictly defined on the mesh of the whole domain in x and z, onto the 1D footprint mesh.

# In[ ]:


fig, axes = plt.subplots()
axes.set_ylabel("thickness (m)")
firedrake.plot(icepack.depth_average(h), axes=axes);


# The velocity profile again reproduces the gradual rise to a high value attained on the ice shelf.

# In[ ]:


fig, axes = plt.subplots()
axes.set_ylabel("ice speed (m / yr)")
firedrake.plot(icepack.depth_average(u), axes=axes);


# Next, we'd like to plot the total mass balance, which in this case is
# 
# $$f = a - \frac{\partial}{\partial x}(hu).$$
# 
# Again, there are some oscillations upstream of the grounding line, but the overall magnitude is much smaller than when we started the simulation.

# In[ ]:


f = icepack.depth_average(firedrake.project(a - (h * u).dx(0), Q))
fig, axes = plt.subplots()
axes.set_ylim(-0.1, +0.1)
axes.set_ylabel("meters / year")
colors = firedrake.plot(f, axes=axes);


# To put everything together, we'll plot the geometry of the equilibrated ice stream, with the bed in brown and the ice base and surface in blue.
# The results are again very similar to that of the ice stream case.

# In[ ]:


fig, axes = plt.subplots()
axes.set_xlabel("distance along centerline (m)")
axes.set_ylabel("elevation (m)")
firedrake.plot(
    icepack.depth_average(b), edgecolor="tab:brown", axes=axes
)
z_b = firedrake.interpolate(s - h, Q)
firedrake.plot(
    icepack.depth_average(z_b), edgecolor="tab:blue", axes=axes
)
firedrake.plot(
    icepack.depth_average(s), edgecolor="tab:blue", axes=axes
);


# ### Resolving shear flow
# 
# Unlike the ice stream model, however, the hybrid model can resolve vertical shear flow.
# To quantity the degree to which the glacier is in plug flow vs shear flow, we can calculate a weighted depth average of the velocity.
# In this case, we'll use the function
# 
# $$q(\zeta) = \sqrt 3 \cdot (2\zeta - 1)$$
# 
# as our weight function.
# This weight is odd when reflected over the axis $\zeta = 1/2$, so it's orthogonal to all plug flow fields.
# The extra factor of $\sqrt 3$ is there to normalize the weight function so that
# 
# $$\int_0^1q(\zeta)^2d\zeta = 1.$$
# 
# This weight is actually just the degree-1 Legendre polynomial, normalized and shifted over to the interval $[0, 1]$.
# 
# We've already seen the `depth_average` function used before to turn fields defined on the full mesh in x and z into fields on the 1D mesh.
# It can also evaluate a weighted averages.
# The weight can be any expression; in our case we're using a polynomial in the symbol $\zeta$, but we could also have used another finite element function.

# In[ ]:


import numpy as np

u_shear = icepack.depth_average(u, weight=np.sqrt(3) * (2 * ζ - 1))


# To wrap things up, let's plot the shear flow mode and do a bit of postprocessing.

# In[ ]:


fig, axes = plt.subplots()
axes.set_title("Shear component of ice velocity")
axes.set_ylabel("velocity (m / yr)")
firedrake.plot(u_shear, axes=axes);


# The region where the velocity has a significant shear flow component is confined to the left half of the domain, where the ice is grounded.
# There is virtually no shear flow at all in the right half of the domain where the ice is floating, just as we expect.
# To be more exact, we can also calculate the unweighted depth average of the velocity and take the ratio of the two components.

# In[ ]:


from firedrake import inner, sqrt

u_avg = icepack.depth_average(u)

U_shear = sqrt(inner(u_shear, u_shear))
U_avg = sqrt(inner(u_avg, u_avg))
ratio = firedrake.interpolate(U_shear / U_avg, u_avg.function_space())


# In[ ]:


fig, axes = plt.subplots()
axes.set_title("Ratio of shear / plug flow")
firedrake.plot(ratio, axes=axes);


# The shear/plug ratio tops out at around 7%.
# For comparison, when the vertical velocity profile follows the shallow ice approximation, the horizontal velocity is a quartic polynomial in the vertical:
# 
# $$u \propto 1 - (1 - \zeta)^4.$$
# 
# In this case, the shear/plug ratio can be calculated by hand to be $\frac{1}{2\sqrt3} \approx 29\%$.
# We could arrive at a different shear ratio in steady state by using a larger value for the basal friction coefficient.

# ### Conclusion
# 
# In this demo, we showed how to use a hybrid model that can capture both plug and shear flow.
# The biggest differences between this model and the ice stream model we showed earlier are that you need to create an extruded mesh and that you need to specify the element family and degree in the vertical dimension.
# Using spectral methods in the vertical dimension makes it possible to capture much of the variation along this axis without resorting to a large number of layers, as might be necessary if we were using a more "traditional" discretization.
# There are also a few extra steps when you want to analyze or plot 3D fields; depth averaging with a good choice of vertical weight function can get you pretty far.
# In principle, this hybrid model will work well on much larger spatial scales such as an entire glacier drainage basin or even a whole ice sheet.

# ### Epilogue: bootstrapping
# 
# When we chose a function space to represent the velocity field, we used a vertical degree of 1, which can resolve linear vertical variations.
# Under the shallow ice approximation, the ice velocity is a quartic polynomial in $\zeta$, so using a degree of at least 4 might have been a more sensible choice.
# 
# The degree-1 solution can still be useful as a cheaper initial guess for the degree-4 solution.
# We rarely have a good initial guess for what the steady-state velocity will be, and computing it from a cold start as we did for the degree-1 solution can be more expensive the more vertical modes we use.
# Instead, we can run the degree-1 velocity and thickness to steady state and use this to initialize a (hopefully cheaper) degree-4 solution.

# In[ ]:


V_2 = firedrake.FunctionSpace(
    mesh, "CG", 2, vfamily="GL", vdegree=2
)
u_2 = firedrake.interpolate(u, V_2)

V_4 = firedrake.FunctionSpace(
    mesh, "CG", 2, vfamily="GL", vdegree=4
)
u_4 = firedrake.interpolate(u, V_4)

solver4 = icepack.solvers.FlowSolver(model, **opts)
u_4 = solver4.diagnostic_solve(
    velocity=u_4,
    thickness=h,
    surface=s,
    fluidity=A,
    friction=C
)


# Observe again how we had to create a new solver, which we called `solver4` here, to calculate the higher-degree velocity field.
# The model object is the same because it's the same physics.
# The solver changes because the solution we're looking for has more degrees of freedom, and thus the nonlinear system characterizing this solution is larger.
# 
# To see how much of a change we get with the extra vertical modes, we'll calculate the relative error in the 2-norm.

# In[ ]:


δu = firedrake.interpolate(u_4 - u_2, V_4)
print(firedrake.norm(δu) / firedrake.norm(u_4))


# A relative error of only 4% is not bad at all!
# However, if we had used a synthetic glacier that was closer to the idealized shallow ice approximation then the error would be larger.
# The accuracy of the degree-2 solution in this particular case should not be extrapolated to all scenarios.
# 
# When we initialize the glacier from a rough initial guess as we did for the degree-2, we have to expect that it will take the entire residence time for any initial transients to propagate out, which in our case is in excess of 100 years.
# If we initialize a degree-4 solution from the steady state degree-2 solution we may be able to get away with a much shorter spin-up period.
