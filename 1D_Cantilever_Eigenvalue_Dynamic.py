#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:29:27 2018

@author: loch
"""
import fenics
import matplotlib.pyplot as plt
import numpy as np

# Define parameters
  
N_eig = 5   # number of eigenvalues
E_ = 2E10
rho_ = 1E4
#rho = fenics.Constant(1.)
L_ = 1 ;  H_ = 0.05 ; W_ = 0.1;  
I_ = W_*H_**3/12.
g = fenics.Constant(0.0) # Displacement at x = 0
theta = fenics.Constant(0.0) # Angle at x = 0
M = fenics.Constant(0.0) # Momentum on the end of the cantilever x = 1
Q = fenics.Constant(0.0) # Shear Force at x = 1
f = fenics.Constant(0.0) # Force across the beam


# Inertia Moment
I = fenics.Constant(I_)
E = fenics.Constant(E_)
rho = fenics.Constant(rho_)
H = fenics.Constant(H_)
W = fenics.Constant(W_)

# Compiler Parameters
fenics.parameters["form_compiler"]["cpp_optimize"] = True
fenics.parameters["form_compiler"]["optimize"] = True
fenics.parameters["ghost_mode"] = "shared_facet"
Nx = 51 

# Create a mesh
mesh = fenics.UnitIntervalMesh(Nx)

# Create a function  space Using Continus Galerkin functions
V = fenics.FunctionSpace(mesh, "Lagrange", 2)
tol = 1E-14

# Define boundary condition
def left(x, on_boundary):
    return on_boundary and fenics.near(x[0], 0, tol)

# Dirichlet
bc = fenics.DirichletBC(V, g, left)

# Define trial and test functions
Phi = fenics.TrialFunction(V) # Trial function
w = fenics.TestFunction(V) # Is this the weigthing function in this case?

##### Neumann
class BoundaryX0(fenics.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-14
            return on_boundary and fenics.near(x[0], 0, tol)


class BoundaryX1(fenics.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-14
            return on_boundary and fenics.near(x[0], 1, tol)
        

boundary_markers = fenics.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)
bx0 = BoundaryX0()
bx1 = BoundaryX1()
bx0.mark(boundary_markers, 1)
bx1.mark(boundary_markers, 2)
ds = fenics.Measure('ds', domain=mesh, subdomain_data=boundary_markers)

for x in mesh.coordinates():
            if bx0.inside(x, True): print('%s is on x = 0' % x)
            if bx1.inside(x, True): print('%s is on x = 1' % x)

# Define normal component, mesh size and right-hand side
h = fenics.CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2.0
n = fenics.FacetNormal(mesh)

alpha = E*I
B =  fenics.inner(fenics.div(fenics.grad(w)),E*I*fenics.div(fenics.grad(Phi)))*fenics.dx \
    - (fenics.jump(fenics.grad(w), n)*E*I*fenics.avg(fenics.div(fenics.grad(Phi))))*fenics.dS \
    - (E*I*fenics.avg(fenics.div(fenics.grad(w)))*fenics.jump(fenics.grad(Phi),n))*fenics.dS \
    + alpha/h_avg*fenics.jump(fenics.grad(w), n)*fenics.jump(fenics.grad(Phi), n)*fenics.dS \
    - fenics.dot(fenics.grad(w) , n)*E*I*fenics.div(fenics.grad(Phi))*ds(1) \
    - E*I*fenics.div(fenics.grad(w))*fenics.dot(fenics.grad(Phi), n)*ds(1) \
    + 2*alpha/h*fenics.dot(fenics.grad(w),n)*fenics.dot(fenics.grad(Phi),n)*ds(1) \

# Assemble stiffness form from Bilinear form
K = fenics.PETScMatrix()
U = fenics.PETScVector()
#l_form = fenics.Constant(1.)*w*fenics.dx
L = fenics.dot(fenics.grad(w) , n)*M/E/I*ds(2) \
+ theta/E/I*fenics.div(fenics.grad(w))*ds(1) + w*Q/E/I*ds(2)  + 2*alpha/h*fenics.dot(fenics.grad(w),n)*theta*ds(1)
fenics.assemble_system(B, L, bc, A_tensor = K, b_tensor = U)
# Assemble mass matrix
M_form = rho*H_*W_*fenics.dot(Phi,w)*fenics.dx
M = fenics.PETScMatrix()
fenics.assemble(M_form, tensor = M)
M = fenics.as_backend_type(M)
K = fenics.as_backend_type(K)
# Solve Eigenvalue problem
eigensolver = fenics.SLEPcEigenSolver(K, M)
eigensolver.parameters['problem_type'] = 'gen_hermitian'
eigensolver.parameters["spectrum"] = "smallest real"
eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
eigensolver.parameters['spectral_shift'] = 0.
eigensolver.parameters["verbose"] = False
eigensolver.parameters['tolerance'] = 1e-14
#eigensolver.parameters['solver']   = 'lapack'
print ("Computing %i first eigenvalues..." % N_eig)
eigensolver.solve(N_eig)

# Exact solution computation - Analytical solution
from scipy.optimize import root
from math import cos, cosh, pi
falpha = lambda x: cos(x)*cosh(x)+1
alpha = lambda n: root(falpha, (2*n+1)*pi/2.)['x'][0]

x = np.linspace(0, L_, Nx)
Dict = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth']
# Extraction and plot
freq_CDC = []
freq_analytical = []
Diff = []
for ii in range(N_eig):
    # Extract eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(ii+1)
    freq_CDC.append(fenics.sqrt(abs(r))/2/fenics.pi)

    # Beam eigenfrequency
    lam = alpha(ii)/L_
    freq_analytical.append(lam**2*np.sqrt(E_*I_/(rho_*H_*W_))/2/pi)
    Diff.append(100*abs(lam**2*np.sqrt(E_*I_/(rho_*H_*W_))/2/pi - fenics.sqrt(abs(r))/2/fenics.pi)/\
                max(lam**2*np.sqrt(E_*I_/(rho_*H_*W_))/2/pi,fenics.sqrt(abs(r))/2/fenics.pi))
    print("\n \n FEM: {0:3.3f} [Hz]   Analytical: {1:3.3f} [Hz]  Diff: {2:3.3f} [%]".format(freq_CDC[ii],freq_analytical[ii], Diff[ii]))

    # Initialize function and assign eigenvector (renormalize by stiffness matrix)
    eigenmode = fenics.Function(V, name="Eigenvector "+str(ii))
    eigenmode.vector()[:] = rx
    plt.figure(ii, figsize = (10,2.5))
    plt.title(str(Dict[ii])+' mode', fontsize = 16)
    if max(abs(eigenmode.compute_vertex_values(mesh))) > abs(min(eigenmode.compute_vertex_values(mesh))):
        fenics.plot(eigenmode/max(abs(eigenmode.compute_vertex_values(mesh))))
    else:
        fenics.plot(-eigenmode/max(abs(eigenmode.compute_vertex_values(mesh))))
    Phi_mode = np.cos(alpha(ii)*x) - np.cosh(alpha(ii)*x) + (np.cos(alpha(ii))+np.cosh(alpha(ii)))/(np.sin(alpha(ii)) + \
                      np.sinh(alpha(ii)))*(np.sinh(alpha(ii)*x) - np.sin(alpha(ii)*x))
    if max(abs(Phi_mode)) > abs(min(Phi_mode)):
        Phi_mode = Phi_mode/max(abs(Phi_mode))
    else:
        Phi_mode = - Phi_mode/max(abs(Phi_mode))
    plt.plot(x/L_, Phi_mode, '-.')
    plt.legend(('C/DG', 'Analytical'), fontsize = 14, ncol = 2, loc = 2)  
    plt.xlabel('x', fontsize = 16)  
    plt.ylabel('$\phi(x)$', fontsize = 16)  
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.hold
    plt.savefig('PDF/'+str(Dict[ii])+'_mode.pdf')
    plt.show()

