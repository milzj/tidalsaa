from dolfin import *
from dolfin_adjoint import *
from collections import OrderedDict
import numpy as np
from .split_samples import SplitSamples

set_log_active(False)

class LocalSAAFunctional(object):

	def __init__(self, random_problem, sampler, sample_indices_rank, number_samples_rank, mpi_rank):
		
		self.random_problem = random_problem
		self.sampler = sampler
		self.number_samples_rank = number_samples_rank
		self.mpi_rank = mpi_rank
		self.sample_indices_rank = sample_indices_rank

	def __call__(self, u):
		"""
		Computes mean over reduced parameterized objective functions

		The mean value is computed using eq. (4) in Assencio (2015).

		Thanks to Gernot Holler for remaining me of that formula.

		References:
		-----------
		D. Assencio, Numerically stable computation of arithmetic means, 2015, blog post,
		https://diego.assencio.com/?index=c34d06f4f4de2375658ed41f70177d59

		R. F. Ling, Comparison of Several Algorithms for Computing Sample Means and Variances,
		Journal of the American Statistical Association
		Vol. 69, No. 348 (Dec., 1974), pp. 859-866
		https://doi.org/10.2307/2286154
		"""
		number_samples_rank = self.number_samples_rank
		mpi_rank = self.mpi_rank

		j = 0.0
		i = 0.0
		#print("number_samples_rank={} mpi_rank {}".format(number_samples_rank,mpi_rank))
		for sample_index in self.sample_indices_rank:
			sample = self.sampler.sample(sample_index)
			#print("sample_index={} mpi_rank {} sample {}".format(sample_index, mpi_rank, sample))
			_j = self.random_problem(u, sample)
			j += 1.0/(i+1.0)*(_j-j)
			i += 1.0

		
		return j


class LocalReducedSAAFunctional(SplitSamples):

	def __init__(self, random_problem, control, sampler, number_samples, mpi_comm = MPI.comm_world):

		self.mpi_comm = mpi_comm
		mpi_size = mpi_comm.Get_size()
		self.mpi_size = mpi_size
		mpi_rank = mpi_comm.Get_rank()
		self.mpi_rank = mpi_rank

		super().__init__(number_samples, mpi_size, mpi_rank)

		local_saa_f = LocalSAAFunctional(random_problem, sampler, self.sample_indices_rank, self.number_samples_rank, mpi_rank)
		local_saa_reduced_rf = ReducedFunctional(local_saa_f(control), Control(control))
		self.rf = local_saa_reduced_rf

	def __call__(self, values):
		return self.rf(values)
	def derivative(self):
		return self.rf.derivative()
	def hessian(self, m_dot):
		return self.rf.hessian(m_dot)

	


