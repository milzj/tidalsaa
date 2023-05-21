

from tidalfarm.random_problem.sampler import Sampler
from scipy.stats import qmc
from scipy.stats import truncnorm

class TidalfarmSampler(Sampler):


    def __init__(self, d=1, m=2, a=0.0, b=1.0, loc=0.0, std=1.0):

        sampler = qmc.Sobol(d=d, scramble=False)
        q = sampler.random_base2(m=m)
        q = q + 1/(2*2**m) # shift

        a, b = (a - loc) / std, (b - loc) / std
        samples = truncnorm.ppf(q, a, b, loc=loc, scale=std)
        self._samples = samples


    def sample(self, sample_index):
            return self._samples[sample_index][0]


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    import os
    path = 'truncated_normal'
    os.makedirs(path, exist_ok=True)


    loc = 0.0025
    a = 0.0
    b = 1.0

    std = 0.01
    m = 20
    sampler = TidalfarmSampler(m=m, a=a, b=b, loc=loc, std=std)
    print(min(sampler._samples))
    print(max(sampler._samples))
    print("------------------")
    print(sampler._samples[0])
    print(sampler.sample(0))


    fig, ax = plt.subplots(1, 1)
    r = sampler._samples
    nbins = 50
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=nbins)
    ax.set_title("truncated normal variables")

    fig.tight_layout()
    plt.savefig(os.path.join(path, "hist_truncated_normal.png"))
