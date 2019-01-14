#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:29:27 2018

@author: loch
"""
import fenics
import matplotlib.pyplot as plt


# Define parameters
E_ = 2E10
L_ = 1 ;  H_ = 0.05 ; W_ = 0.1;  
I_ = W_*H_**3/12.
Nx = 200 

# Inertia Moment
I = fenics.Constant(I_)
E = fenics.Constant(E_)
H = fenics.Constant(H_)
W = fenics.Constant(W_)

g = fenics.Constant(0.00) # Displacement at x = 0
theta = fenics.Constant(0.0) # Angle at x = 0
M = fenics.Constant(1.0) # Momentum on the end of the cantilever x = 1
Q = fenics.Constant(0.0) # Shear Force at x = 1
f = fenics.Constant(0.0) # Force across the beam

# Next, some parameters for the form compiler are set:
# Optimization options for the form compiler
fenics.parameters["form_compiler"]["cpp_optimize"] = True
fenics.parameters["form_compiler"]["optimize"] = True
# Make mesh ghosted for evaluation of DG terms
fenics.parameters["ghost_mode"] = "shared_facet"

# Create a mesh
mesh = fenics.UnitIntervalMesh(Nx)

# Create a function  space Using Continus Galerkin functions
V = fenics.FunctionSpace(mesh, "Lagrange", 4)
tol = 1E-14

# Define boundary condition
def left(x, on_boundary):
    return on_boundary and fenics.near(x[0], 0, tol)

# Define trial function and Test Function
bc = fenics.DirichletBC(V, g, left)
Phi = fenics.TrialFunction(V) 
w = fenics.TestFunction(V) 

##### Neumann
class BoundaryX0(fenics.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-14
            return on_boundary and fenics.near(x[0], 0, tol)

class BoundaryX1(fenics.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-14
            return on_boundary and fenics.near(x[0], 1, tol)
            
################ FacetFunction doesn't exist, maybe MeshFunction works 
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
# Penalty parameter
alpha = E*I
# Bilinear form
B = fenics.inner(fenics.div(fenics.grad(w)),E*I*fenics.div(fenics.grad(Phi)))*fenics.dx \
- (fenics.jump(fenics.grad(w), n)*E*I*fenics.avg(fenics.div(fenics.grad(Phi))))*fenics.dS \
- (E*I*fenics.avg(fenics.div(fenics.grad(w)))*fenics.jump(fenics.grad(Phi),n))*fenics.dS \
+ alpha/h_avg*fenics.jump(fenics.grad(w), n)*fenics.jump(fenics.grad(Phi), n)*fenics.dS \
- fenics.dot(fenics.grad(w) , n)*E*I*fenics.div(fenics.grad(Phi))*ds(1) \
- E*I*fenics.div(fenics.grad(w))*fenics.dot(fenics.grad(Phi), n)*ds(1) \
+ 2*alpha/h*fenics.dot(fenics.grad(w),n)*fenics.dot(fenics.grad(Phi),n)*ds(1)

# And the linear form
L = w*f*fenics.dx \
+ fenics.dot(fenics.grad(w) , n)*M*ds(2) \
+ E*I*theta*fenics.div(fenics.grad(w))*ds(1) \
- w*Q*ds(2) \
- 2*alpha/h*fenics.dot(fenics.grad(w),n)*theta*ds(1)

# Solve the variational problem
Phi = fenics.Function(V)
fenics.solve(B == L, Phi, bc)

plt.figure(figsize=(10,8))
fenics.plot(mesh, title="Finite element mesh")

if g.values() == 0 and theta.values() == 0 and f.values() == 0 and Q.values() == 0:
    Phi_Expr= fenics.Expression('0.5*M/E/I*x[0]*x[0]', degree = 2, M = M, E = E, I = I)
    Analytical = True
elif g.values() == 0 and theta.values() == 0 and f.values() == 0 and M.values() == 0:
    Phi_Expr = fenics.Expression('0.5*Q/E/I*(0.3333*x[0]*x[0]*x[0] - x[0]*x[0])', degree = 3, Q = Q, E = E, I = I)
    Analytical = True
elif g.values() == 0 and theta.values() == 0 and Q.values() == 0 and M.values() == 0:
    Phi_Expr = fenics.Expression('0.041666666666666664*f/E/I*(pow(x[0],4) -4*pow(x[0],3) + 6*x[0]*x[0])', degree = 4, f = f, E = E, I = I)    
    Analytical = True
else:
    Analytical = False
if Analytical:        
    Phi_Analytical = fenics.project(Phi_Expr, V)

plt.figure(figsize = (10,2.5))
plt.title('Displacement', fontsize = 16)
fenics.plot(Phi)
if Analytical:
    fenics.plot(Phi_Analytical, linestyle = '-.')
plt.legend(('C/DG', 'Analytical'), fontsize = 14)  
plt.xlabel('x', fontsize = 16)  
plt.ylabel('$\phi$', fontsize = 16)  
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('PDF/Static_Displacement.pdf')

plt.figure(figsize = (10,2.5))
plt.title('Angle across the cantilever', fontsize = 16)
fenics.plot(Phi.dx(0))
if Analytical:
    fenics.plot(Phi_Analytical.dx(0), linestyle = '-.')
plt.legend(('C/DG', 'Analytical'), fontsize = 14)  
plt.xlabel('x', fontsize = 16)  
plt.ylabel('$\partial \phi/ \partial x$', fontsize = 16)  
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('PDF/Static_Angle.pdf')

plt.figure(figsize = (10,2.5))
plt.title('Moment across the cantilever $(M = E\, I\, \partial^2 \phi/ \partial x^2)$', fontsize = 16)
fenics.plot(E*I*fenics.div(fenics.grad(Phi)))
if Analytical:
    fenics.plot((E_*I_*Phi_Analytical.dx(0)).dx(0), linestyle = '-.')
plt.legend(('C/DG', 'Analytical'), fontsize = 14)  
plt.xlabel('x', fontsize = 16)  
plt.ylabel('$M$', fontsize = 16)  
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('PDF/Static_Moment.pdf')

plt.figure(figsize = (10,2.5))
plt.title('Shear Force across the cantilever $(Q = E\, I\, \partial^3 \phi/ \partial x^3)$', fontsize = 16)
fenics.plot(E*I*fenics.div(fenics.grad(Phi)).dx(0))
if Analytical:
    fenics.plot(((Phi_Analytical.dx(0)).dx(0)).dx(0), linestyle = '-.')
plt.legend(('C/DG', 'Analytical'), fontsize = 14)  
plt.xlabel('x', fontsize = 16)  
plt.ylabel('$Q$', fontsize = 16)  
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('PDF/Static_Shear_Force.pdf')
plt.show()

