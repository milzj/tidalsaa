import os, sys


import numpy as np
from dolfin import *
from dolfin_adjoint import *

from random_tidalfarm_problem import RandomTidalfarmProblem
from tidalfarm.problem.riesz_map import RieszMap
from tidalfarm.solver_options import SolverOptions
from tidalfarm.random_problem import GlobalReducedSAAFunctional

from tidalfarm_sampler import TidalfarmSampler

from random_field import RandomField

import moola
import fw4pde
import matplotlib.pyplot as plt



if MPI.comm_world.Get_rank() == 0:

    date = sys.argv[1]
    m = int(sys.argv[2])
    a = float(sys.argv[3])
    b = float(sys.argv[4])
    std = float(sys.argv[5])
    loc = float(sys.argv[6])

    _outdir = "output/" + date + "/"

    if not os.path.exists(_outdir):
        os.makedirs(_outdir)

else:
    date = None
    _outdir = None
    m = None
    a = None
    b = None
    std = None
    loc = None


date = MPI.comm_world.bcast(date, root=0)
outdir = MPI.comm_world.bcast(_outdir, root=0)
m = MPI.comm_world.bcast(m, root=0)
a = MPI.comm_world.bcast(a, root=0)
b = MPI.comm_world.bcast(b, root=0)
std = MPI.comm_world.bcast(std, root=0)
loc = MPI.comm_world.bcast(loc, root=0)


random_problem = RandomTidalfarmProblem()
control_space = random_problem.control_space
control = Function(control_space)

# Objective function
beta = random_problem.beta
lb = random_problem.lb
ub = random_problem.ub
scaled_L1_norm = fw4pde.problem.ScaledL1Norm(control_space,beta)
bottom_friction = 0.016
J = random_problem(control, bottom_friction)

ctrl = Control(control)


number_samples = 2**m
#sampler = TidalfarmSampler(d=1, m=m, loc=loc, a=a, b=b, std=std)
sampler = RandomField(scale=0.05)
sampler.function_space = control_space

global_saa_rf = GlobalReducedSAAFunctional(random_problem, control, sampler, number_samples)


# Optimization problem
problem = MoolaOptimizationProblem(global_saa_rf)
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
n = random_problem.tidal_problem.domain_mesh.n
plt.set_cmap("coolwarm")
c = plot(solution_final)
plt.colorbar(c)
plt.savefig(outdir + "solution_final_n_{}.pdf".format(n), bbox_inches="tight")
plt.savefig(outdir + "solution_final_n_{}.png".format(n), bbox_inches="tight")
plt.close()

solution_best = sol["control_best"].data
u_vec = solution_best.vector()[:]
scaling = 100.0/max(u_vec)
_control = Function(control_space)
_control.vector()[:] = scaling*u_vec
plt.set_cmap("coolwarm")
c = plot(_control)
plt.gca().set_xlabel("meters")
plt.gca().set_ylabel("meters")
cb = plt.colorbar(c, label="Turbine density", shrink=1, orientation="horizontal")
# https://stackoverflow.com/questions/34458949/matplotlib-colorbar-formatting
cb.ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
plt.savefig(outdir + "solution_best_n_{}.pdf".format(n), bbox_inches="tight")
plt.savefig(outdir + "solution_best_n_{}.png".format(n), bbox_inches="tight")
plt.close()
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
site_dx = random_problem.tidal_problem.domain_mesh.site_dx
cost_coefficient = random_problem.tidal_problem.cost_coefficient
thrust_coefficient = random_problem.tidal_problem.tidal_parameters.thrust_coefficient.values()[0]
turbine_cross_section = random_problem.tidal_problem.tidal_parameters.turbine_cross_section.values()[0]

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











