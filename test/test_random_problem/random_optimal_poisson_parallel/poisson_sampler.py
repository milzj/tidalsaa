from random_problem import Sampler
import numpy as np

class PoissonSampler(Sampler):

	def __init__(self, grid_points=10):

		x = np.linspace(3.0, 5.0, grid_points)
		y = np.linspace(0.5, 2.5, grid_points)
		xx, yy = np.meshgrid(x,y)
		self.xx = np.ndarray.flatten(xx)
		self.yy = np.ndarray.flatten(yy)

	def sample(self, sample_index):
		return [self.xx[sample_index], self.yy[sample_index]]

if __name__ == "__main__":

	grid_points = 10
	sampler = PoissonSampler(grid_points)

	assert grid_points**2 == len(sampler.xx)

	sample_indices = [0, 44, 99]
	for sample_index in sample_indices:
		sample = sampler.sample(sample_index)
		print(sample)
