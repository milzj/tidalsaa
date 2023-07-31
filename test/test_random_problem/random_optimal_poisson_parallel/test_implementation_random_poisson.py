import pytest

from dolfin import *
from dolfin_adjoint import *
import numpy as np

from .random_poisson_problem import RandomPoissonProblem
from .poisson_sampler import PoissonSampler


@pytest.mark.parametrize("sample_index", [0, 12, 24])
def test_implementation_random_poisson(sample_index):

	set_working_tape(Tape())

	n = 32
	random_problem = RandomPoissonProblem(n=32)

	grid_points = 5
	sampler = PoissonSampler(grid_points=grid_points)

	U = random_problem.U
	u = Function(U)
	u_expr = Expression("sin(pi*x[0])*sin(pi*x[1])", degree = 3)
	u.interpolate(u_expr)

	h = Function(U)
	h_expr = Expression("exp(pi*x[0])*cos(pi*x[1])", degree = 3)
	h.interpolate(h_expr)

	sample = sampler.sample(sample_index)
	J = random_problem(u, sample)
	rf = ReducedFunctional(J, Control(u))

	conv_rate = taylor_test(rf, u, h)

	assert np.isclose(conv_rate, 2.0, atol=0.2)
