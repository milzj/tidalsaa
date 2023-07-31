import pytest

from dolfin import *
from dolfin_adjoint import *
import moola

import numpy as np

set_log_level(30)

def convergence_rates(E_values, eps_values, show=True):
	"""
	Source https://github.com/dolfin-adjoint/pyadjoint/blob/master/pyadjoint/verification.py
	"""
	from numpy import log
	r = []
	for i in range(1, len(eps_values)):
		r.append(log(E_values[i] / E_values[i - 1])
		/ log(eps_values[i] / eps_values[i - 1]))
	if show:
		print("Computed convergence rates: {}".format(r))
	return r

def random_optimal_poisson(n, ydm=1.0, k=2, alpha=1e-3, samples=np.ones(1), gtol=1e-9):
	"""Defines a simple risk-neutral optimal control problem.

	The diffusion coefficient is a random scalar. The SAA solution for the infinite
	dimensional control problem is known (called u_analytic below).

	The problem is partly based on that considered in
	http://www.dolfin-adjoint.org/en/latest/documentation/poisson-mother/poisson-mother.html
	"""
	set_working_tape(Tape())

	mesh = UnitSquareMesh(n, n)

	V = FunctionSpace(mesh, "CG", 1)
	W = FunctionSpace(mesh, "DG", 0)
	bc = DirichletBC(V, 0.0, "on_boundary")


	yd = Expression("m*sin(k*pi*x[0])*sin(k*pi*x[1])", degree = 3, m = ydm, k=k)
	u = Function(W)
	y = Function(V)
	v = TestFunction(V)

	m1 = np.mean(1/samples)
	m2 = np.mean(1/samples**2)

	um = (2.0*k**2*np.pi**2*ydm*m1) / (4.0*k**4*np.pi**4*alpha+m2)

	u_analytic = Expression("m*sin(k*pi*x[0])*sin(k*pi*x[1])", degree = 3, m = um, k=k)

	# compute sample average
	J = 0.
	for i in range(len(samples)):

		F = Constant(samples[i])*inner(grad(y), grad(v))*dx - u * v * dx
		solve(F == 0, y, bc)

		j = assemble(0.5 * inner(y - yd, y - yd) * dx + .5*Constant(alpha)*u**2*dx)
		J += 1.0/(i+1.0)*(j-J)

	rf = ReducedFunctional(J, Control(u))
	problem = MoolaOptimizationProblem(rf)
	u_moola = moola.DolfinPrimalVector(u)

	solver = moola.NewtonCG(problem, u_moola, options={'gtol': gtol,
                                                   'maxiter': 20,
                                                   'display': 3,
                                                   'ncg_hesstol': 0})

	sol = solver.solve()
	u_opt = sol['control'].data

	control_error = errornorm(u_analytic, u_opt, degree_rise=3)

	return control_error

def test_random_optimal_poisson_convergence_rates():
	"""Checks if the theoretical convergence rate equals the empirical rate.

	The theoretical convergence rate is O(h^2). We observe a first oder convergence.
	"""
	# problem data

	ydm = 1.0/(2*np.pi**2)
	k = 2
	alpha = 1e-3

	N = 10
	np.random.seed(1234)
	samples = np.exp(np.random.randn(N))
	gtol = 1e-9

	errors = []
	values = []

	for n in [8, 16, 32, 64, 128]:

		error = random_optimal_poisson(n, ydm=ydm, k=k, alpha=alpha, samples=samples, gtol=gtol)
		errors.append(error)
		values.append(1.0/n)

	rate = np.median(convergence_rates(errors, values))
	assert rate > 1.0

