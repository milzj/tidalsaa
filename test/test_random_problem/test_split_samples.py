import pytest

from random_problem import SplitSamples

def test_split_samples():

	number_samples = 11
	mpi_size = 3

	split_samples = SplitSamples(number_samples, mpi_size, 0)

	assert 4 == split_samples.number_samples_rank

	split_samples = SplitSamples(number_samples, mpi_size, 2)

	assert 3 == split_samples.number_samples_rank


	number_samples = 3
	mpi_size = 11
	with pytest.raises(ValueError):
		split_samples = SplitSamples(number_samples, mpi_size, 0)
