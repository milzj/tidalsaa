from dolfin import *
from dolfin_adjoint import *

#set_log_level(30)

from random_problem import RandomProblem

class RandomPoissonProblem(RandomProblem):
	"""
	We divide the domain into two equally sized subdomains. The
	random diffusion coefficient modelling material parameters is
	constant on each subdomain. The implementation of the diffusion
	coefficient is based on that in the FENICS tutorial volume 1
	(see https://fenicsproject.org/pub/tutorial/html/._ftut1013.html)

	References:
	----------

	H. P. Langtangen and A. Logg, Solving PDEs in Python: The FEniCS tutorial I,
	Simula SpringerBriefs Comput. 3, Springer, Cham, 2016,
	https://doi.org/10.1007/978-3-319-52462-7.

	S. W. Funke: http://www.dolfin-adjoint.org/en/latest/documentation/poisson-mother/poisson-mother.html
	"""

	def __init__(self, n):

		self.n = n

		mesh = UnitSquareMesh(MPI.comm_self, n, n)
		V = FunctionSpace(mesh, "CG", 1)
		U = FunctionSpace(mesh, "DG", 0)

		self.V = V
		self.U = U

		self.y = Function(V)
		self.v = TestFunction(V)

		self.u = Function(U)

		self._alpha = Constant(1e-3)

		self.yd = Expression("1.0/(2.0*pi*pi)*sin(pi*x[0])*sin(pi*x[1])", \
				degree = 3, mpi_comm=mesh.mpi_comm())

		self.bcs = DirichletBC(self.V, 0.0, "on_boundary")

		tol = 1E-14
		p1 = 1.0
		p2 = 0.01

		self.kappa = Expression('x[1] <= 0.5 + tol ? p1 : p2', degree=0,\
               			tol=tol, p1=p1, p2=p2, mpi_comm=mesh.mpi_comm())


	@property
	def alpha(self):
		return self._alpha

	@property
	def control_space(self):
		return self.U


	def state(self, y, v, u, params):

		self.kappa.p1 = params[0]
		self.kappa.p2 = params[1]

		F = (self.kappa*inner(grad(y), grad(v)) - u * v) * dx
		solve(F == 0, y, bcs=self.bcs)

	def __call__(self, u, sample):

		y = self.y
		yd = self.yd
		params = sample
		y.vector().zero()
		v = self.v
		alpha = self.alpha
		self.state(y, v, u, params)

		return 0.5 * assemble((y-yd) ** 2 * dx + (alpha/2) *u**2*dx)


