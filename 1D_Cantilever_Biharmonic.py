#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:29:27 2018

@author: loch
"""
import fenics
import matplotlib.pyplot as plt
#from fenics import *

Two_Dimensional = False

# Define parameters
E = fenics.Constant(1.)
I = fenics.Constant(1.)
L = 1. ;  H = 0.1 ;  Nx = 200 ; Ny = 4
# Boundary conditions
g = fenics.Constant(0.0) # Displacement at x = 0
theta = fenics.Constant(0.0) # Angle at x = 0
M = fenics.Constant(0.0) # Momentum on the end of the cantilever x = 1
Q = fenics.Constant(1.0) # Shear Force at x = 1
f = fenics.Constant(0.0) # Force across the beam

# Penalty parameter
alpha = E*I

# Next, some parameters for the form compiler are set:
# Optimization options for the form compiler
fenics.parameters["form_compiler"]["cpp_optimize"] = True
fenics.parameters["form_compiler"]["optimize"] = True
# Make mesh ghosted for evaluation of DG terms
fenics.parameters["ghost_mode"] = "shared_facet"

# Create a mesh
if Two_Dimensional:
    mesh = fenics.RectangleMesh(fenics.Point(0., 0.), fenics.Point(L, H), Nx, Ny, "crossed")
else:
    mesh = fenics.UnitIntervalMesh(Nx)

# Create a function  space Using Continus Galerkin functions
V = fenics.FunctionSpace(mesh, "Lagrange", 5)
tol = 1E-14

# Define boundary condition
def left(x, on_boundary):
    return on_boundary and fenics.near(x[0], 0, tol)

# Dirichlet
bc = fenics.DirichletBC(V, g, left)

##### Neumann
class BoundaryX0(fenics.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-14
            return on_boundary and fenics.near(x[0], 0, tol)
#            return on_boundary and fenics.near(x[0], 0, 5e11*tol)

class BoundaryX1(fenics.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-14
#            return x[0] >= 0.5 + tol
#            return fenics.near(x[0], 1, 5e11*tol)
            return on_boundary and fenics.near(x[0], 1, tol)
            
################ FacetFunction doesn't exist, maybe MeshFunction works 
boundary_markers = fenics.MeshFunction('size_t', mesh, mesh.topology().dim())
#boundary_markers.set_all(2)
bx0 = BoundaryX0()
bx1 = BoundaryX1()
bx0.mark(boundary_markers, 0)
bx1.mark(boundary_markers, 1)

ds = fenics.Measure('ds', domain=mesh, subdomain_data=boundary_markers)

for x in mesh.coordinates():
            if bx0.inside(x, True): print('%s is on x = 0' % x)
            if bx1.inside(x, True): print('%s is on x = 1' % x)


# Define trial and test functions
Phi = fenics.TrialFunction(V) # Trial function
w = fenics.TestFunction(V) # Is this the weigthing function in this case?

# Define normal component, mesh size and right-hand side
h = fenics.CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2.0
n = fenics.FacetNormal(mesh)
 
B = fenics.inner(fenics.div(fenics.grad(w)),E*I*fenics.div(fenics.grad(Phi)))*fenics.dx \
- (fenics.jump(fenics.grad(w), n)*E*I*fenics.avg(fenics.div(fenics.grad(Phi))))*fenics.dS \
- (E*I*fenics.avg(fenics.div(fenics.grad(w)))*fenics.jump(fenics.grad(Phi),n))*fenics.dS \
+ alpha/h_avg*fenics.jump(fenics.grad(w), n)*fenics.jump(fenics.grad(Phi), n)*fenics.dS \
- fenics.dot(fenics.grad(w) , n)*E*I*fenics.div(fenics.grad(Phi))*ds(0) \
- E*I*fenics.div(fenics.grad(w))*fenics.dot(fenics.grad(Phi), n)*ds(0) \
+ 2*alpha/h*fenics.dot(fenics.grad(w),n)*fenics.dot(fenics.grad(Phi),n)*ds(0)

# And the linear form
L = w*f*fenics.dx \
+ fenics.dot(fenics.grad(w) , n)*M*ds \
+ E*I*theta*fenics.div(fenics.grad(w))*ds(0) \
- w*Q*ds \
- 2*alpha/h*fenics.dot(fenics.grad(w),n)*theta*ds(0)


# Solve the variational problem
Phi = fenics.Function(V)
fenics.solve(B == L, Phi, bc)

plt.figure(figsize=(10,8))
fenics.plot(mesh, title="Finite element mesh")

plt.figure()
fenics.plot(boundary_markers, title="Subdomain x = 0")
fenics.plot(mesh)
plt.xlim(-0.01, 0.1)

plt.figure()
fenics.plot(boundary_markers, title="Subdomain x = 1")
fenics.plot(mesh)
plt.xlim(0.9, 1.01)

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
        
Phi_Analytical = fenics.project(Phi_Expr, V)

plt.figure(figsize = (10,3))
fenics.plot(Phi, title = 'Displacement')
if Analytical:
    fenics.plot(Phi_Analytical, linestyle = '-.')
plt.legend(('C/DG', 'Analytical'))    
plt.xlim(0, 1)
plt.figure(figsize = (10,3))
fenics.plot(Phi.dx(0), title = 'Angle across the cantilever')
if Analytical:
    fenics.plot(Phi_Analytical.dx(0), linestyle = '-.')
plt.legend(('C/DG', 'Analytical'))    
plt.xlim(0, 1)
plt.figure(figsize = (10,3))
fenics.plot(fenics.div(fenics.grad(Phi)), title = 'Moment across the cantilever')
if Analytical:
    fenics.plot((Phi_Analytical.dx(0)).dx(0), linestyle = '-.')
plt.legend(('C/DG', 'Analytical'))   
plt.xlim(0, 1)
plt.figure(figsize = (10,3))
fenics.plot(fenics.div(fenics.grad(Phi)).dx(0), title = 'Shear Force across the cantilever')
if Analytical:
    fenics.plot(((Phi_Analytical.dx(0)).dx(0)).dx(0), linestyle = '-.')
plt.legend(('C/DG', 'Analytical'))   
plt.ylim(-2, 2)
plt.xlim(0., 1)
plt.show()




