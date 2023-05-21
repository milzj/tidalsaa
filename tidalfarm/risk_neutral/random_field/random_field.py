import numpy as np
import itertools
from fenics import *
from dolfin_adjoint import *

class RandomField(object):

	def __init__(self, mean=0.5, l=0.5, m=100):
		"""
		Implements the random field used in sect. 4.2 in Ref. [1]

		References:
		----------
		[1] C. Geiersbach and T. Scarinci, Stochastic proximal gradient methods for nonconvex
		    problems in Hilbert spaces, Comput. Optim. Appl., 78 (2021), pp. 705–740, https:
		    //doi.org/10.1007/s10589-020-00259-y.

			Arguments:
			----------
				mean : float
					scalar mean
				l : float
					correlation length
				m : int
					number of addens
		"""
		self.version = 1000

		self.mean = mean
		self.l = l
		self.m = m

		indices = list(itertools.product(range(1,self.m), repeat=2))
		indices = sorted(indices, key=lambda x:x[0]**2+x[1]**2)
		self.indices = indices[0:self.m]

	def bump_seed(self):
		self.version += 1

	@property
	def function_space(self):
		return self._function_space

	@function_space.setter
	def function_space(self, function_space):
		self._function_space = function_space
		self.eigenpairs()

	def eigenpairs(self):
		"""Computes eigenvalues and eigenfunctions."""

		function_space = self.function_space

		eigenvalues = np.zeros(len(self.indices))
		eigenfunctions = []

		for i, pair in enumerate(self.indices):     
			j, k = pair[0], pair[1]

			eigenvalue = .25*np.exp(-np.pi*(j**2+k**2)*self.l**2)
			eigenvalues[i] = eigenvalue

			fun = Expression("2.0*cos(pi*j*x[0])*cos(pi*k*x[1])", j=j, k=k, degree=1)

			eigenfunction = interpolate(fun, function_space)
			eigenfunctions.append(eigenfunction)

		self.eigenvalues = eigenvalues
		self.eigenfunctions = eigenfunctions

	def plot_eigenvalues(self, outdir):

		import matplotlib.pyplot as plt

		m = self.m
		e = self.eigenvalues
		plt.scatter(range(m), e, s=0.5)
		plt.xlabel("Index of eigenvalue")
		plt.ylabel("Eigenvalue")
		plt.yscale("log")

		plt.tight_layout()
		plt.savefig(outdir + "log_eigenvalues.pdf")


	def mean_field(self):

		V = self.function_space

		return interpolate(Constant(self.mean), V)

	def sample(self):
		return self.realization()

	def realization(self):
		"""Computes a realization of the random field.

		The seed is increased by one.

		Note:
		-----
		The sum defining the truncated KL expansion is evaluated from smallest
		to largest eigenvalue.
		"""
		self.bump_seed()
		np.random.seed(self.version)
		xi = np.random.uniform(-np.sqrt(0.5), np.sqrt(0.5), self.m)

		y = self.mean_field().vector().get_local()

		for i in np.argsort(self.eigenvalues):

			value = self.eigenvalues[i]
			fun = self.eigenfunctions[i]

			y += np.sqrt(value)*xi[i]*fun.vector().get_local()


		f = Function(self.function_space)
		f.vector()[:] = y

		return f
