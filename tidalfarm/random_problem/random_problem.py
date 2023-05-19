
class RandomProblem(object):

	def __call__(self, u, sample):

		raise NotImplementedError("RandomProblem.__call__ is not implemented.")

	@property
	def control_space(self):
		raise NotImplementedError("RandomProblem.control_space is not implemented.")

