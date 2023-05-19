

from tidalfarm.random_problem.sampler import Sampler
from scipy.stats import qmc
from scipy.stats import truncnorm

class TidalfarmSampler(Sampler):


    def __init__(self, d=1, m=2, a=0.0, b=1.0, loc=0.0, std=1.0):

        sampler = qmc.Sobol(d=d, scramble=False)
        q = sampler.random_base2(m=m)
        q = q + 1/(2*2**m) # shift

        samples = truncnorm.ppf(q, a/std, b/std, loc=loc, scale=std)
        self._samples = samples


    def sample(self, sample_index):
            return self._samples[sample_index][0]


if __name__ == "__main__":

    loc = 0.005
    sampler = TidalfarmSampler(m=3, loc=loc, std=5.0*loc)
    print(sampler._samples)
    print("------------------")
    print(sampler._samples[0])
    print(sampler.sample(0))
