import numpy as np
from dolfin import *
from dolfin_adjoint import *

class Experiments(object):

    def __init__(self):

        self._experiments = {}

        name = "Bottom_Friction_Viscosity"
        # Bottom friction also called (quadratic) bed roughness coefﬁcient
        # or natural bed friction coefficient:
        # compare with eq. (3.1) in Kreitmair (2021)

        # pp. 5 and 8 in Kreitmair et al. (2019), https://doi.org/10.1098/rsos.180941
        # p. 104 in Kreitmair (2021), https://doi.org/10.1007/978-3-030-57658-5
        bottom_friction_vec = [Constant(0.0025), Constant(0.005), Constant(0.016)]

        # p. 54 in Kreitmair (2021), https://doi.org/10.1007/978-3-030-57658-5
        bottom_friction_vec.append(Constant(0.001))
        bottom_friction_vec.append(Constant(0.0035))

        viscosities = [1.0, 2.0, 3.0, 4.0, 5.0]

        viscosity_vec = []

        for viscosity in viscosities:
                viscosity_vec.append(Constant(viscosity))

        b_vec = []
        v_vec = []

        for b in bottom_friction_vec:

            for v in viscosity_vec:

                b_vec.append(b)
                v_vec.append(v)


        self.add_experiment(name, b_vec, v_vec)


    def add_experiment(self, experiment_name, bottom_friction_vec, viscosity_vec):

        key = ("bottom_friction", "viscosity")
        items = list(list(zip(np.array(bottom_friction_vec),np.array(viscosity_vec))))

        print("Number of experiments: {}".format(len(items)))

        self._experiments[experiment_name] = {key: items}


    def __call__(self, experiment_name):


        return self._experiments[experiment_name]



if __name__ == "__main__":

    experiments = Experiments()

    experiment_name = "Bottom_Friction_Viscosity"

    experiment = experiments(experiment_name)
    print(experiment[("bottom_friction", "viscosity")][0])
