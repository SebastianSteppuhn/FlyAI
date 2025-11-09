import numpy as np
import gmsh, meshio
from mpi4py import MPI
from dolfinx import mesh, fem, io
import ufl

# -------------------------
# Geometry: backward-facing step
# Channel length L=6, downstream height H=2, upstream height h=1
# Step located at x=1 (jump from h to H). Units arbitrary.
# -------------------------
L, H, h = 6.0, 2.0, 1.0
inlet_x, outlet_x, step_x = 0.0, L, 1.0

gmsh.initialize()
gmsh.model.add("bfs")
p = gmsh.model.occ

# Downstream block (big channel)
rect1 = p.addRectangle(step_x, 0, 0, L - step_x, H)
# Upstream block (small channel, height h)
rect2 = p.addRectangle(inlet_x, 0, 0, step_x - inlet_x, h)
# Fuse to create the step shape
p.fragment([(2, rect1)], [(2, rect2)])
p.synchronize()

# Mark physical groups for BCs
# Get curves (1D entities) on boundaries
curves = gmsh.model.getEntities(dim=1)

# Helper to find boundary by coordinate match
def near(val, target, tol=1e-8): return abs(val - target) < tol

inlet, outlet, walls = [], [], []
for (dim, cid) in curves:
    com = gmsh.model.occ.getCenterOfMass(dim, cid)
    x, y, _ = com
    # identify inlet and outlet by x center
    if near(x, 0.5 * step_x, tol=1e-6):      # inlet side (x in [0,1])
        inlet.append(cid)
    elif near(x, (L + step_x) * 0.5, tol=1e-6):  # outlet side (x in [1,6])
        outlet.append(cid)
    else:
        walls.append(cid)

ps_inlet  = gmsh.model.addPhysicalGroup(1, inlet, name="inlet")
ps_outlet = gmsh.model.addPhysicalGroup(1, outlet, name="outlet")
ps_walls  = gmsh.model.addPhysicalGroup(1, walls, name="walls")

# 2D physical group for the fluid
surfs = [e[1] for e in gmsh.model.getEntities(dim=2)]
gmsh.model.addPhysicalGroup(2, surfs, name="fluid")

# Mesh size: coarser=larger number
gmsh.model.mesh.generate(2)
gmsh.write("bfs.msh")
gmsh.finalize()

# Convert to XDMF/VTK-friendly and load into dolfinx
msh = meshio.read("bfs.msh")
# Extract cell/physical data
triangle_cells = [c for c in msh.cells if c.type == "triangle"][0].data
line_cells = [c for c in msh.cells if c.type == "line"][0].data

cell_data = {}
for name, data in msh.cell_data_dict["gmsh:physical"].items():
    cell_data[name] = [data[msh.cells.index(next(c for c in msh.cells if c.type == "triangle"))]]
line_data = {}
for name, data in msh.cell_data_dict["gmsh:physical"].items():
    line_data[name] = [data[msh.cells.index(next(c for c in msh.cells if c.type == "line"))]]

meshio.write(
    "bfs_mesh.xdmf",
    meshio.Mesh(points=msh.points,
                cells=[("triangle", triangle_cells)],
                cell_data={k: [cell_data[k][0]] for k in cell_data if k in cell_data}),
    file_format="xdmf"
)
meshio.write(
    "bfs_facet.xdmf",
    meshio.Mesh(points=msh.points,
                cells=[("line", line_cells)],
                cell_data={k: [line_data[k][0]] for k in line_data if k in line_data}),
    file_format="xdmf"
)

# Load into dolfinx
with io.XDMFFile(MPI.COMM_WORLD, "bfs_mesh.xdmf", "r") as xf:
    mshx = xf.read_mesh(name="Grid")
mshx.name = "mesh"

with io.XDMFFile(MPI.COMM_WORLD, "bfs_facet.xdmf", "r") as ff:
    mt = ff.read_meshtags(mshx, name="Grid")

# Get facet tags
inlet_id  = next(tag.value for tag in mt.values if tag.name == "inlet")
outlet_id = next(tag.value for tag in mt.values if tag.name == "outlet")
walls_id  = next(tag.value for tag in mt.values if tag.name == "walls")

# -------------------------
# Function spaces and parameters
# -------------------------
V = fem.VectorFunctionSpace(mshx, ("Lagrange", 2))
Q = fem.FunctionSpace(mshx, ("Lagrange", 1))

u  = fem.Function(V, name="u")
v  = ufl.TestFunction(V)
p  = fem.Function(Q, name="p")
q  = ufl.TestFunction(Q)

u_prev = fem.Function(V)

rho = 1.0
nu  = 1e-3  # kinematic viscosity
Umax = 1.0  # peak inlet speed (adjust to set Reynolds number)

# Inlet profile: parabolic on the local inlet height (0..h)
y = ufl.SpatialCoordinate(mshx)[1]
uin = fem.Expression(ufl.as_vector([4*Umax*(y*(h - y))/h**2, 0]), V.element.interpolation_points())
u_inlet = fem.Function(V)
u_inlet.interpolate(uin)

# Boundary conditions
facet_inlet  = mesh.locate_entities(mshx, 1, lambda x: np.isclose(x[0], inlet_x))
facet_outlet = mesh.locate_entities(mshx, 1, lambda x: np.isclose(x[0], outlet_x))
# Use meshtags from Gmsh (more robust than coordinate checks)
inlet_facets  = mt.find(inlet_id)
outlet_facets = mt.find(outlet_id)
walls_facets  = mt.find(walls_id)

bc_inlet = fem.dirichletbc(u_inlet, fem.locate_dofs_topological(V, 1, inlet_facets))
bc_walls = fem.dirichletbc(fem.Function(V), fem.locate_dofs_topological(V, 1, walls_facets))  # zero vector = no-slip
bcs_u = [bc_inlet, bc_walls]

# Weak forms (steady NSE with Picard linearization around u_prev)
def nlin(u_):  # (u_prev · ∇) u_
    return ufl.dot(ufl.grad(u_), u_prev)

f = ufl.as_vector((0.0, 0.0))
a_mom = rho*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx*nu + rho*ufl.inner(nlin(u), v)*ufl.dx - ufl.div(v)*p*ufl.dx
L_mom = ufl.inner(f, v)*ufl.dx

a_cont = ufl.inner(ufl.div(u), q)*ufl.dx
L_cont = ufl.Constant(mshx, 0.0)*q*ufl.dx

# PETSc solvers
problem_mom = fem.petsc.NonlinearProblem(a_mom + a_cont, fem.Function(V), bcs=bcs_u, J=None)
# We'll instead assemble two linear problems: velocity with fixed pressure, then pressure Poisson, but for brevity we do a simple mixed solve:
W = fem.FunctionSpace(mshx, ufl.MixedElement(V.ufl_element(), Q.ufl_element()))
(U, P) = ufl.TrialFunctions(W)
(Vt, Qt) = ufl.TestFunctions(W)

a = (rho*nu*ufl.inner(ufl.grad(U), ufl.grad(Vt)) + rho*ufl.inner(ufl.dot(ufl.grad(U), u_prev), Vt) - ufl.div(Vt)*P + Qt*ufl.div(U))*ufl.dx
L = ufl.inner(f, Vt)*ufl.dx

w = fem.Function(W)
(u_, p_) = w.sub(0), w.sub(1)

# Build Dirichlet BCs on mixed space
dofs_u = fem.locate_dofs_topological(W.sub(0), 1, inlet_facets)
bc_inlet_m = fem.dirichletbc(u_inlet, dofs_u, W.sub(0))
dofs_w = fem.locate_dofs_topological(W.sub(0), 1, walls_facets)
bc_walls_m = fem.dirichletbc(fem.Function(V), dofs_w, W.sub(0))
bcs = [bc_inlet_m, bc_walls_m]

problem = fem.petsc.LinearProblem(a, L, bcs=bcs)

# Picard loop
for k in range(20):
    u_prev.x.array[:] = u_.collapse().x.array if k > 0 else 0.0
    w = problem.solve()
    u_.x.scatter_forward()
    p_.x.scatter_forward()
    # crude convergence check
    if MPI.COMM_WORLD.rank == 0:
        print(f"Picard iter {k+1}")

# Save results
with io.XDMFFile(MPI.COMM_WORLD, "bfs_results.xdmf", "w") as xdmf:
    xdmf.write_mesh(mshx)
    uh = fem.Function(V, name="u");  uh.x.array[:] = u_.collapse().x.array
    ph = fem.Function(Q, name="p");  ph.x.array[:] = p_.collapse().x.array
    xdmf.write_function(uh)
    xdmf.write_function(ph)

if MPI.COMM_WORLD.rank == 0:
    print("Wrote bfs_results.xdmf (open in ParaView).")
