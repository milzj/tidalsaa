from dolfin import *
from dolfin_adjoint import *

#set_log_level(30)

from tidalfarm.random_problem import RandomProblem

from tidalfarm.problem.domain_farm import DomainFarm
from tidalfarm.problem.tidal_parameters import TidalParameters
from tidalfarm.problem.tidal_problem import TidalProblem
from tidalfarm.problem.riesz_map import RieszMap


class RandomTidalfarmProblem(RandomProblem):


    def __init__(self):

        tidal_parameters = TidalParameters()
        domain_farm = DomainFarm()
        self.tidal_problem = TidalProblem(tidal_parameters, domain_farm)

        self._beta = self.tidal_problem.beta
        self._lb = self.tidal_problem.lb
        self._ub = self.tidal_problem.ub

    @property
    def control_space(self):
        return self.tidal_problem.control_space

    @property
    def beta(self):
        return self._beta

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    def __call__(self, control, bottom_friction_value):

        self.tidal_problem.tidal_parameters.bottom_friction = bottom_friction_value

        return self.tidal_problem(control)


