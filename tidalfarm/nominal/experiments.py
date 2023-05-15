import numpy as np
from dolfin import *
from dolfin_adjoint import *

class Experiments(object):

    def __init__(self):

        self._experiments = {}

        name = "Bottom_Friction"
        # Bottom friction also called (quadratic) bed roughness coefﬁcient
        # or natural bed friction coefficient:
        # compare with eq. (3.1) in Kreitmair (2021)

        # pp. 5 and 8 in Kreitmair et al. (2019), https://doi.org/10.1098/rsos.180941
        # p. 104 in Kreitmair (2021), https://doi.org/10.1007/978-3-030-57658-5
        bottom_friction_vec = [Constant(0.0025), Constant(0.005), Constant(0.016)]

        # p. 54 in Kreitmair (2021), https://doi.org/10.1007/978-3-030-57658-5
        bottom_friction_vec.append(Constant(0.001))
        bottom_friction_vec.append(Constant(0.0035))


        self._experiments[name] = {"bottom_friction": bottom_friction_vec}


    def __call__(self, experiment_name):

        return self._experiments[experiment_name]



if __name__ == "__main__":

    experiments = Experiments()

    experiment_name = "Bottom_Friction"

    e = experiments(experiment_name)
    print(e["bottom_friction"])
