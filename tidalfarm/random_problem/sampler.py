import numpy as np

class Sampler(object):
	"""A generic class to implement a sampler."""

	def sample(self, sample_index):
		"""Computes and returns a sample.

		Parameters:
		-----------
		sample_index : int or ndarray
			A number used to set a seed.
		"""
		raise NotImplementedError("Sampler.sample is not implemented.")

