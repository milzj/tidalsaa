import os, sys


import numpy as np
from dolfin import *
from dolfin_adjoint import *

from tidalfarm.problem.domain_farm import DomainFarm
from tidalfarm.problem.tidal_parameters import TidalParameters
from tidalfarm.problem.tidal_problem import TidalProblem
from tidalfarm.problem.riesz_map import RieszMap
from tidalfarm.solver_options import SolverOptions

import moola
import fw4pde
import matplotlib.pyplot as plt

set_log_level(30)

from experiments import Experiments

experiments = Experiments()
experiment_name = "Bottom_Friction"



print(MPI.comm_world.Get_rank())
print(MPI.comm_world.Get_size())

if MPI.comm_world.Get_rank() == 0:
    date = sys.argv[1]
    _outdir = "output/" + experiment_name + "/" + date + "/"

    if not os.path.exists(_outdir):
        os.makedirs(_outdir)

else:
    date = None
    _outdir = None

date = MPI.comm_world.bcast(date, root=0)
_outdir = MPI.comm_world.bcast(_outdir, root=0)

bottom_frictions = experiments(experiment_name)["bottom_friction"]
mpi_rank = MPI.comm_world.Get_rank()

bottom_friction = bottom_frictions[mpi_rank]
bottom_friction_value = bottom_friction.values()[0]

# Tidal farm problem
tidal_parameters = TidalParameters()
tidal_parameters.bottom_friction = bottom_friction
viscosity_value = tidal_parameters.viscosity.values()[0]

outdir = _outdir + "bottom_friction_{}_viscosity_{}/".format(bottom_friction_value, viscosity_value)

if not os.path.exists(outdir):
    os.makedirs(outdir)

print("-"*40)
print("Bottom friction = {}".format(bottom_friction_value))
print("-"*40)


domain_farm = DomainFarm()
tidal_problem = TidalProblem(tidal_parameters, domain_farm)
control_space = tidal_problem.control_space
control = Function(control_space)

# Objective function
beta = tidal_problem.beta
lb = tidal_problem.lb
ub = tidal_problem.ub
scaled_L1_norm = fw4pde.problem.ScaledL1Norm(control_space,beta)
ctrl = Control(control)
J = tidal_problem(control)
rf = ReducedFunctional(J, ctrl)

# Optimization problem
problem = MoolaOptimizationProblem(rf)
riesz_map = RieszMap(control_space)
u_moola = moola.DolfinPrimalVector(control, riesz_map = riesz_map)
u_moola.zero()

box_constraints = fw4pde.problem.BoxConstraints(control_space, lb, ub)
moola_box_lmo = fw4pde.algorithms.MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

# Solver
solver_options = SolverOptions()
options = solver_options.options
stepsize = solver_options.stepsize

solver = fw4pde.algorithms.FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,\
                stepsize=stepsize, lmo=moola_box_lmo, options=options)

sol = solver.solve()

# Postprocessing: Plotting and saving
solution_final = sol["control_final"].data
n = tidal_problem.domain_mesh.n
plt.set_cmap("coolwarm")
c = plot(solution_final)
plt.colorbar(c)
plt.savefig(outdir + "solution_final_n_{}.pdf".format(n))
plt.savefig(outdir + "solution_final_n_{}.png".format(n))
plt.close()

solution_best = sol["control_best"].data
c = plot(solution_best)
plt.colorbar(c)
plt.savefig(outdir + "solution_best_n_{}.pdf".format(n))
plt.savefig(outdir + "solution_best_n_{}.png".format(n))
plt.close()

filename = outdir + "solution_best_n={}.txt".format(n)
np.savetxt(filename, solution_best.vector()[:])
filename = outdir + "solution_final_n={}.txt".format(n)
np.savetxt(filename, solution_final.vector()[:])


file = File(outdir + "solution" +  "_best_n={}".format(n) + ".pvd")
file << solution_best
file = File(outdir + "solution" +  "_final_n={}".format(n) + ".pvd")
file << solution_final


# Compute and print results (source: https://zenodo.org/record/224251)
site_dx = tidal_problem.domain_mesh.site_dx
cost_coefficient = tidal_problem.cost_coefficient
thrust_coefficient = tidal_parameters.thrust_coefficient.values()[0]
turbine_cross_section = tidal_parameters.turbine_cross_section.values()[0]

site_area = assemble(Constant(1.0)*site_dx(1))
power = -problem.obj(sol["control_best"])
total_friction = assemble(solution_best*site_dx(1))
cost = cost_coefficient * total_friction
friction = 0.5*thrust_coefficient*turbine_cross_section # see model_turbine.py https://zenodo.org/record/224251

with open(outdir + "postprocessing.txt", "a") as f:

    print("="*40, file=f)
    print("Site area (m^2): {}".format(site_area), file=f)
    print("Cost coefficient: {}".format(cost_coefficient), file=f)
    print("Total power: %e" % power, file=f)
    print("Total cost: %e" % cost, file=f)
    print("Total turbine friction: %e" % total_friction, file=f)
    print("Average smeared turbine friction: %e" % (total_friction / site_area), file=f)
    print("Total power / total friction: %e" % (power / total_friction), file=f)
    print("Friction per discrete turbine: {}".format(friction), file=f)
    print("Estimated number of discrete turbines: {}".format(total_friction/friction), file=f)
    print("Estimated average power per turbine: {}".format(power / (total_friction/friction)), file=f)











