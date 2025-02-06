# Janel Chua :D
# Remember to activate fenics in the terminal before running
#############################################################################
# Preliminaries and mesh
from dolfin import *
import numpy as np
mesh = Mesh('Trial18.xml')

# Introduce manually the material parameters
class GC(UserExpression):
    def set_Gc_values(self, Gc_0, Gc_1):
        self.Gc_0, self.Gc_1 = Gc_0, Gc_1
    def eval(self, value, x):
        "Set value[0] to value at point x"
        tol = 1E-14
        if x[1] >= 0.015 + tol:
            value[0] = self.Gc_0
        elif x[1] <= -0.015 + tol:
            value[0] = self.Gc_0
        else:
            value[0] = self.Gc_1 #middle layer
# Initialize Gc
Gc = GC()
Gc.set_Gc_values(0.081, 0.00001) # N/mm

l = 0.015 # mm
E  = 5.3*1000 # N/mm^2
nu = 0.35 # Poisson's ratio
mu    = Constant(E / (2.0*(1.0 + nu))) # N/mm^2
lmbda = Constant(E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))) # N/mm^2

# Mass density
rho = Constant(1.23*10**(-9)) # (N.s^2)/mm^4

# Viscous stress parameter
eta_m = Constant(0)
eta_k = Constant(1e-8)

eta_e = Constant(1e-6)

# Generalized-alpha method parameters
alpha_m = Constant(0.2)
alpha_f = Constant(0.4)
gamma   = Constant(0.5+alpha_f-alpha_m)
beta    = Constant((gamma+0.5)**2/4.)

# Time-stepping parameters
T       = .0002 
Nsteps  = 100000 # each timestep is 2e-9s
dt = Constant(T/Nsteps)

# Define traction (if that is the boundary condition)
tract = Constant((10,0.0)) # N/mm^2

#############################################################################
V = FunctionSpace(mesh, 'CG', 1)
p, q = TrialFunction(V), TestFunction(V)
pnew = Function(V)
pnewTemp = Function(V)
pnew2 = Function(V)
pold = Function(V)
# Setting an initial value for phi
class InitialCondition(UserExpression):
    def eval_cell(self, value, x, ufc_cell):
        if abs(x[1]) < 5e-03 and x[0] <= -0.25:
            value[0] = 1.0
        else:
            value[0] = 0.0
pold.interpolate(InitialCondition())

W = VectorFunctionSpace(mesh, 'CG', 1)
WW = FunctionSpace(mesh, 'DG', 0)
u, v_ = TrialFunction(W), TestFunction(W)
# Current (unknown) displacement
unew = Function(W)
unewTemp = Function(W)
# Fields from previous time step (displacement, velocity, acceleration)
uold = Function(W)
vold = Function(W)
aold = Function(W)

#############################################################################
# Boundary conditions
def top(x,on_boundary):
    return near(x[1],1) and on_boundary #(x[0], 1)
def bot(x,on_boundary):
    return near(x[1],-1) and on_boundary #(x[0], -1)

def left(x,on_boundary):
    return near(x[0],-0.5) and on_boundary #(-0.5, x[1])
def leftTopHalf(x,on_boundary):
    return near(x[0],-0.5) and (x[1] > 0) and on_boundary #(-0.5, x[1])
def leftBotHalf(x,on_boundary):
    return near(x[0],-0.5) and (x[1] < 0) and on_boundary #(-0.5, x[1])
def right(x,on_boundary):
    return near(x[0],5) and on_boundary #(5, x[1])

def leftcorner(x,on_boundary):
    tol=1E-15
    return (abs(x[0]+0.5) < tol) and (abs(x[1]+1)<tol) #(-0.5,-1)
def rightcorner(x,on_boundary):
    tol=1E-15
    return (abs(x[0]-5) < tol) and (abs(x[1]+1)<tol) #(5,-1)

loadtop = Expression("t", t = 0.0, degree=1)
loadbot = Expression("t", t = 0.0, degree=1)
loadleft = Expression("t", t = 0.0, degree=1)
bcbot= DirichletBC(W, Constant((0.0,0.0)), bot) # Bottom fixed in both x and y
bcleft1 = DirichletBC(W.sub(1), Constant(0), left) # u1=0, left boundary not allowed to move vertically
bcleft2 = DirichletBC(W.sub(0), loadleft, leftTopHalf) # leftTopHalf displacement loaded
bcleft3 = DirichletBC(W.sub(0), Constant(0), leftBotHalf) # u0=0 on leftBotHalf no horizontal displacement

bc_u = [bcbot, bcleft1, bcleft2, bcleft3]
bc_phi = []
n = FacetNormal(mesh)

# Create mesh function over the cell facets
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
AutoSubDomain(top).mark(boundary_subdomains, 1)
# Define measure for boundary condition integral
dss = ds(subdomain_data=boundary_subdomains)

#############################################################################
# Constitutive functions
def epsilon(u):
    return sym(grad(u))
def sigma(u):
    return 2.0*mu*epsilon(u)+lmbda*tr(epsilon(u))*Identity(len(u))
def psi(u):
    return 0.5*lmbda*(tr(epsilon(u)))**2 + mu*inner(epsilon(u),epsilon(u)) # isotropic linear elasticity
# def psi_pos(u):
#     return 0.5*(lmbda+mu)*(0.5*(tr(epsilon(u))+abs(tr(epsilon(u)))))**2 + mu*inner(dev(epsilon(u)),dev(epsilon(u)))
# def psi_neg(u):
#     return 0.5*(lmbda+mu)*(0.5*(tr(epsilon(u))-abs(tr(epsilon(u)))))**2
# def dev_psi_pos(u):
#     return (lmbda+mu)*(0.5*(tr(epsilon(u))+abs(tr(epsilon(u)))))*Identity(len(u))+2*mu*dev(epsilon(u))
# def dev_psi_neg(u):
#     return (lmbda+mu)*(0.5*(tr(epsilon(u))-abs(tr(epsilon(u)))))*Identity(len(u))
# def sigma2(pold,u):
#     return ((1.0-pold)**2)*(dev_psi_pos(u)) + (dev_psi_neg(u))  

# Mass form
def m(unew, v_):
    return rho*inner(unew, v_)*dx
# Elastic stiffness form
def k(pold, unew, v_):
    return ((1.0-pold)**2 + eta_e)*inner(sigma(unew), sym(grad(v_)))*dx  
# Rayleigh damping form
def c(pold,unew, v_):
    return eta_m*m(unew, v_) + eta_k*k(pold, unew, v_)
# Work of external forces
def Wext(pold, v_):
    return ((1.0-pold)**2 + eta_e)*dot(v_, tract)*dss(1)
def H_l(x):
    return (1/2)*(1 + tanh(x/l))
def preventHeal(pold,pnew):
    coor = mesh.coordinates()
    p_cr = 0.995
    pnew_nodal_values = pnew.vector()
    pnew_array = pnew_nodal_values.get_local()
    pold_nodal_values = pold.vector()
    pold_array = pold_nodal_values.get_local()
    for i in range(len(pold_array)):
        if pnew_array[i] < pold_array[i]:
            pnew_array[i] = pold_array[i]
        elif pold_array[i] > p_cr and pnew_array[i] >= pold_array[i]:
            pnew_array[i] = pnew_array[i]
        elif pold_array[i] <= p_cr:
            pnew_array[i] = pnew_array[i]
    #Reverse the projection
    pnew2.vector()[:] = pnew_array[:]   
    return pnew2

# Implicit Newmark Method
# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(unew, uold, vold, aold, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
    return (unew-uold-dt_*vold)/beta_/dt_**2 - (1-2*beta_)/2/beta_*aold

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, uold, vold, aold, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return vold + dt_*((1-gamma_)*aold + gamma_*a)

def update_fields(unew, uold, vold, aold):
    """Update fields at the end of each time step."""
    # Get vectors (references)
    u_vec, u0_vec  = unew.vector(), uold.vector()
    v0_vec, a0_vec = vold.vector(), aold.vector()
    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    # Update (u_old <- u)
    vold.vector()[:], aold.vector()[:] = v_vec, a_vec
    uold.vector()[:] = unew.vector()		

def avg(xold, xnew, alpha):
    return alpha*xold + (1-alpha)*xnew

#############################################################################
# Weak form for momentum balance
anew = update_a(u, uold, vold, aold, ufl=True)
vnew = update_v(anew, uold, vold, aold, ufl=True)
res = m(avg(aold, anew, alpha_m), v_) + c(pold, avg(vold, vnew, alpha_f), v_) \
       + k(pold, avg(uold, u, alpha_f), v_) #- Wext(pold, v_) # uncomment for traction boundary condition
a_form = lhs(res)
L_form = rhs(res)

# Weak form for quasistatic phi-evolution
E_phi = (Gc*l*inner(grad(p),grad(q))+((Gc/l)+2.0*psi(unew))\
            *inner(p,q)-2.0*psi(unew)*q)*dx
a_phi = lhs(E_phi)
L_phi = rhs(E_phi)

#############################################################################
# Initialization of the iterative procedure and output requests
time = np.linspace(0, T, Nsteps+1)
u_r = 5000 # mm
tol = 1e-3

store_phi = File ("mydata/phi.pvd")
store_u = File ("mydata/u.pvd")
store_vel = File ("mydata/vel.pvd")

sigma_fs = TensorFunctionSpace(mesh, "CG", 1)
stress_total = Function(sigma_fs, name='Stress')
store_stress_total = File ("mydata/stress_total.pvd")
stress_elas = Function(sigma_fs, name='Stress')
store_stress_elas = File ("mydata/stress_elas.pvd")
stress_vis = Function(sigma_fs, name='Stress')
store_stress_vis = File ("mydata/stress_vis.pvd")

poynting = VectorFunctionSpace(mesh, "CG", 1)
poynting_temp = Function(poynting, name='poynting')
store_poynting = File ("mydata/poynting.pvd")

#############################################################################
# Looping through time here.
for (i, dt) in enumerate(np.diff(time)):
    t = time[i+1]
    print("Time: ", t)
    if t <= (8/4)*T:
        loadleft.t=t*u_r 
        print ('displacement', loadleft.t)
    else:
        loadtop.t=loadtop.t + (-0.1*u_r) 
        print ('displacement', loadtop.t)
    iter = 0
    err = 1

    while err > tol:
        iter += 1
        # Solve for new displacement    
        solve(a_form == L_form, unew, bc_u, solver_parameters={'linear_solver': 'mumps'})
        solve(a_phi == L_phi, pnew, bc_phi, solver_parameters={'linear_solver': 'mumps'})
        # Prevent healing
        pnew2 = preventHeal(pold,pnew)
        # Calculate error
        err_u = errornorm(unew,unewTemp,norm_type = 'l2',mesh = None)
        err_phi = errornorm(pnew2,pold,norm_type = 'l2',mesh = None)
        err = max(err_u,err_phi)
        # Update new fields in same timestep with new calculated quantities
        unewTemp.vector()[:] = unew.vector()
        pold.assign(pnew2)
        print ('Iterations:', iter, ', Total time', t, ', error', err)
	
        if err < tol:
            # Update old fields from previous timestep with new quantities
            update_fields(unew, uold, vold, aold)
            pold.assign(pnew2)
            print ('err<tol :D','Iterations:', iter, ', Total time', t, ', error', err)

            if round(t*1e9) % 20 == 0: # save data points every 2e-8s
                store_phi << pold
                store_u << uold # mm
                store_vel << vold # mm/s
                
                stress_total.assign(project(sigma(uold)*((1.0-pold)**2 + eta_e) + eta_k*sigma(vold)*((1.0-pold)**2 + eta_e),sigma_fs, solver_type="cg", preconditioner_type="amg")) # 1MPa = 1N/mm^2
                store_stress_total << stress_total
                stress_elas.assign(project(sigma(uold)*((1.0-pold)**2 + eta_e) + 0*eta_k*sigma(vold)*((1.0-pold)**2 + eta_e),sigma_fs, solver_type="cg", preconditioner_type="amg")) # 1MPa = 1N/mm^2
                store_stress_elas << stress_elas
                stress_vis.assign(project(0*sigma(uold)*((1.0-pold)**2 + eta_e) + eta_k*sigma(vold)*((1.0-pold)**2 + eta_e),sigma_fs, solver_type="cg", preconditioner_type="amg")) # 1MPa = 1N/mm^2
                store_stress_vis << stress_vis   
                poynting_temp.assign(project((sigma(uold)*((1.0-pold)**2 + eta_e)+ eta_k*sigma(vold)*((1.0-pold)**2 + eta_e))*vold,poynting, solver_type="cg", preconditioner_type="amg")) # units are in N/(mm.s)
                store_poynting << poynting_temp

                File('mydata/saved_mesh.xml') << mesh
                File('mydata/saved_phi.xml') << pold
                File('mydata/saved_u.xml') << uold
                File('mydata/saved_v.xml') << vold
                File('mydata/saved_a.xml') << aold

                print ('Iterations:', iter, ', Total time', t, 'Saving datapoint')
 	    
print ('Simulation completed') 
#############################################################################
