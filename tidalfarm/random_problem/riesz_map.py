"""
The file adapts RieszMap
https://github.com/funsim/moola/blob/79ffc8d86dff383762063970358b8aebfe373896/moola/adaptors/dolfin_vector.py
and passes the function space's MPI communicator to the LUSolver.
"""

class RieszMap(object):

    def __init__(self, V, inner_product="L2", map_operator=None, inverse = "default"):
        self.V = V
        import dolfin

        if inner_product != "custom":

            u = dolfin.TrialFunction(V)
            v = dolfin.TestFunction(V)

            if isinstance(V, dolfin.cpp.function.MultiMeshFunctionSpace):
                default_forms = {"L2":   dolfin.inner(u, v)*dolfin.dX,
                                 "H0_1": dolfin.inner(dolfin.grad(u), dolfin.grad(v))*dolfin.dX,
                                 "H1":  (dolfin.inner(u, v) + dolfin.inner(dolfin.grad(u),
                                                                           dolfin.grad(v)))*dolfin.dX,
                }
            else:
                default_forms = {"L2":   dolfin.inner(u, v)*dolfin.dx,
                                 "H0_1": dolfin.inner(dolfin.grad(u), dolfin.grad(v))*dolfin.dx,
                                 "H1":  (dolfin.inner(u, v) + dolfin.inner(dolfin.grad(u),
                                                                           dolfin.grad(v)))*dolfin.dx,
                }

            form = default_forms[inner_product]
            if hasattr(form.arguments()[0], "_V_multi"):
                map_operator = dolfin.assemble_multimesh(form)
            else:
                map_operator = dolfin.assemble(form)
        self.map_operator = map_operator
        if inverse in ("default", "lu"):
            mpi_comm = self.V.mesh().mpi_comm()
            self.map_solver = dolfin.LUSolver(mpi_comm, self.map_operator)

        elif inverse == "jacobi":
            self.map_solver = dolfin.PETScKrylovSolver()
            self.map_solver.set_operator(self.map_operator)
            self.map_solver.ksp().setType("preonly")
            self.map_solver.ksp().getPC().setType("jacobi")

        elif inverse == "sor":
            self.map_solver = dolfin.PETScKrylovSolver()
            self.map_solver.set_operator(self.map_operator)
            self.map_solver.ksp().setType("preonly")
            self.map_solver.ksp().getPC().setType("sor")

        elif inverse == "amg":
            self.map_solver = dolfin.PETScKrylovSolver()
            self.map_solver.set_operator(self.map_operator)
            self.map_solver.ksp().setType("preonly")
            self.map_solver.ksp().getPC().setType("hypre")

        elif isinstance(inverse, dolfin.GenericMatrix):
            self.map_solver = dolfin.PETScKrylovSolver()
            self.map_solver.set_operators(self.map_operator, inverse)
            self.map_solver.ksp().setType("preonly")
            self.map_solver.ksp().getPC().setType("mat")

        else:
            self.map_solver = inverse
        self.solver_type = inverse
    def primal_map(self, x, b):
        self.map_solver.solve(x, b)

    def dual_map(self, x):
        return self.map_operator * x

