from dolfin import *
from random_field import RandomField
import matplotlib.pyplot as plt
import sys
import numpy as np

from tidalfarm.risk_neutral.random_tidalfarm_problem import RandomTidalfarmProblem

import os
outdir = "output/"
if not os.path.exists(outdir):
	os.makedirs(outdir)


N = 128 # number of samples

random_problem = RandomTidalfarmProblem()
control_space = random_problem.control_space

rf = RandomField(scale=0.05)
rf.function_space = control_space

rf.plot_eigenvalues(outdir)


for j in range(N):

    u = rf.sample(j)

    filename = "random_field_seed_" + str(j)

    c = plot(u)
    cb = plt.colorbar(c, label="Magnitude", shrink=1, orientation="horizontal")

    plt.savefig(outdir + filename + ".pdf", bbox_inches="tight")
    plt.savefig(outdir + filename + ".png", bbox_inches="tight")
    plt.close()

