import numpy as np


class SplitSamples(object):
	"""
	Uses np.array_split to compute the number of samples per process.
	If the number of samples is smaller than `mpi_size`, an error is
	thrown.

	Parameters:
	-----------
	number_samples	: int
		Total number of samples
	mpi_size	: int
		Total number of processes
	mpi_rank	: int
		MPI rank

	"""

	def __init__(self, number_samples, mpi_size, mpi_rank):

		samples_indices = np.array_split(range(number_samples), mpi_size)

		if mpi_size != np.count_nonzero([arr.size for arr in samples_indices]):
			raise ValueError("mpi_size = {} is larger than \
				the number of samples = {}".format(mpi_size, number_samples))

		number_samples_rank = len(samples_indices[mpi_rank])

		self.number_samples_rank = number_samples_rank
		self.number_samples = number_samples
		self.sample_indices_rank = samples_indices[mpi_rank]
