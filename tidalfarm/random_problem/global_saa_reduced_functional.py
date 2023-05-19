import fenics

import numpy as np
from .local_saa_functional import LocalReducedSAAFunctional

class GlobalReducedSAAFunctional(object):

	def __init__(self, random_problem, control, sampler, number_samples):
		"""
		Before calling Allreduce, we multiply the local sample averges by the number of local
		samples. Thereby, we obtain the sum of the local samples. After reducing, we divide
		the sum by the total number of samples.
		"""

		local_saa_rf = LocalReducedSAAFunctional(random_problem, control, sampler, number_samples)

		self.mpi_comm = local_saa_rf.mpi_comm
		self.number_samples_rank = local_saa_rf.number_samples_rank
		self.number_samples = number_samples
		self.sample_indices_rank = local_saa_rf.sample_indices_rank
		
		self.control_space = control.function_space()
		self.dim_control_space = self.control_space.dim()
		self.rf = local_saa_rf

	def __call__(self, values):

		rf_value = self.rf(values)
		#print("mpi_rank {} rf_value = {}".format(self.mpi_rank, rf_value))
		func_value = np.array(self.number_samples_rank*rf_value, dtype=np.float64)

		#print("mpi_rank {} func_value {}".format(self.mpi_comm.Get_rank(), func_value))
		sum_func_value = np.array(0.0, dtype=np.float64)
		self.mpi_comm.Allreduce(func_value, sum_func_value)
		#print("mpi_rank {} sum_func_value {}".format(self.mpi_comm.Get_rank(), sum_func_value))

		return sum_func_value/self.number_samples

	def derivative(self):

		local_derivative = self.rf.derivative()

		der_array = np.array(self.number_samples_rank*local_derivative.vector().get_local())
		sum_der_array = np.empty(self.dim_control_space, dtype=np.float64)
		self.mpi_comm.Allreduce(der_array, sum_der_array) # MPI.SUM

		global_derivative = fenics.Function(self.control_space)
		global_derivative.vector().set_local(sum_der_array/self.number_samples)

		return global_derivative

	def hessian(self, m_dot):

		local_hessian = self.rf.hessian(m_dot)

		der_array = np.array(self.number_samples_rank*local_hessian.vector().get_local())
		sum_der_array = np.empty(self.dim_control_space, dtype=np.float64)
		self.mpi_comm.Allreduce(der_array, sum_der_array) # MPI.SUM

		global_hessian = fenics.Function(self.control_space)

		global_hessian.vector().set_local(sum_der_array/self.number_samples)

		return global_hessian
