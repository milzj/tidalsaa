import pytest

from dolfin import *
from dolfin_adjoint import *
import moola
import numpy as np

from .random_poisson_problem import RandomPoissonProblem
from .poisson_sampler import PoissonSampler
from random_problem import LocalReducedSAAFunctional, GlobalReducedSAAFunctional
from random_problem import RieszMap

def serial_poisson_problem(n=32, grid_points=7, gtol=1e-9):

	set_working_tape(Tape())

	random_problem = RandomPoissonProblem(n=32)

	sampler = PoissonSampler(grid_points=grid_points)

	number_samples = grid_points**2

	u = Function(random_problem.control_space)

	local_saa_rf = LocalReducedSAAFunctional(random_problem, u, sampler, number_samples, mpi_comm=MPI.comm_self)


	problem = MoolaOptimizationProblem(local_saa_rf)
	riesz_map = RieszMap(random_problem.control_space)
	u_moola = moola.DolfinPrimalVector(u, riesz_map = riesz_map)

	solver = moola.NewtonCG(problem, u_moola, options={'gtol': gtol,
	                                                   'maxiter': 20,
	                                                   'display': 3,
	                                                   'ncg_hesstol': 0})

	sol = solver.solve()

	return sol['control'].data

def parallel_poisson_problem(n=32, grid_points=7, gtol=1e-9):

	set_working_tape(Tape())

	random_problem = RandomPoissonProblem(n=32)

	sampler = PoissonSampler(grid_points=grid_points)

	number_samples = grid_points**2

	u = Function(random_problem.control_space)

	global_saa_rf = GlobalReducedSAAFunctional(random_problem, u, sampler, number_samples)

	problem = MoolaOptimizationProblem(global_saa_rf)
	riesz_map = RieszMap(random_problem.control_space)
	u_moola = moola.DolfinPrimalVector(u, riesz_map = riesz_map)

	solver = moola.NewtonCG(problem, u_moola, options={'gtol': gtol,
	                                                   'maxiter': 20,
	                                                   'display': 3,
	                                                   'ncg_hesstol': 0})

	sol = solver.solve()

	return sol['control'].data


@pytest.mark.parametrize("n", [4, 8])
@pytest.mark.parametrize("grid_points", [5, 10])
def test_serial_parallel_poisson_problem(n, grid_points):

	gtol = 1e-9
	rtol = 1e-12

	u_serial = serial_poisson_problem(n=n, grid_points=grid_points, gtol=gtol)
	u_parallel = parallel_poisson_problem(n=n, grid_points=grid_points, gtol=gtol)

	rerror = errornorm(u_serial, u_parallel, degree_rise = 0)/norm(u_serial)
	print("relative error = {}".format(rerror))
	assert rerror <= 1e-12



if __name__ == "__main__":

	n = 16
	grid_points = 10
	test_serial_parallel_poisson_problem(n, grid_points)

