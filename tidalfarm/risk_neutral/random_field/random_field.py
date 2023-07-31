# original source: https://github.com/milzj/SAA4PDE/blob/consistency/simulations/semilinear_consistency/random_field.py

import numpy as np
import itertools
from fenics import *
from dolfin_adjoint import *

def smoothed_plus(x, mu):
    if x >= 0.0:
        return x+mu*np.logaddexp(0.0, -x/mu)
    else:
        return mu*np.logaddexp(x/mu, 0.0)

class RandomField(object):

    def __init__(self, number_samples = 1000, mean=0.025, l=0.5, m=100, mu = 0.025, scale=0.05):
        """
        Implements a modification random field used in sect. 4.2 in Ref. [1]

        TODO:
        - Point out modifications
        - Implement evaluations via matrix vector multiplication (will result in significant
        speed up)

        References:
        ----------
        [1] C. Geiersbach and T. Scarinci, Stochastic proximal gradient methods for nonconvex
            problems in Hilbert spaces, Comput. Optim. Appl., 78 (2021), pp. 705–740, https:
            //doi.org/10.1007/s10589-020-00259-y.

            Arguments:
            ----------
                mean : float
                    scalar mean
                l : float
                    correlation length
                m : int
                    number of addens
                mu : float
                    smoothing parameter
        """
        self.version = 1000

        self.mean = mean
        self.l = l
        self.m = m
        self.mu = mu

        # https://www.wolframalpha.com/input?i=sum+sqrt%28.25%2F0.5%5E2%29%2F%28k%5E2*l%5E2%29+for+k%3D1..infty+and+l%3D1..infty
        
        self.random_field_max = np.pi**4/36.0
        self.scale = scale

        indices = list(itertools.product(range(1,self.m), repeat=2))
        indices = sorted(indices, key=lambda x:x[0]**2+x[1]**2)
        self.indices = indices[0:self.m]

        self.indices = indices[0:self.m]

        xi_mat = []
        for i in range(number_samples):
            self.bump_seed()
            np.random.seed(self.version)
            xi = np.random.uniform(-1.0, 1.0, self.m)
            xi_mat.append(xi)

        self.xi_mat = xi_mat

    def bump_seed(self):
        self.version += 1

    @property
    def function_space(self):
        return self._function_space

    @function_space.setter
    def function_space(self, function_space):
        self._function_space = function_space
        self.eigenpairs()

    def eigenpairs(self):
        """Computes eigenvalues and eigenfunctions."""

        function_space = self.function_space
        mesh = function_space.mesh()
        mpi_comm = mesh.mpi_comm()

        eigenvalues = np.zeros(len(self.indices))
        eigenfunctions = []

        _addends = []

        for i, pair in enumerate(self.indices):
            j, k = pair[0], pair[1]
            eigenvalue = .25/((j**2+k**2)*self.l**2)
            eigenvalue = .25/((j**2*k**2)*self.l**2)
            eigenvalues[i] = eigenvalue

            fun = Expression("2.0*cos(pi*j*x[0]/1000)*cos(pi*k*x[1]/2000)", j=j, k=k, degree=0, domain=mesh, mpi_comm = mpi_comm)

            eigenfunction = interpolate(fun, function_space)
            eigenfunctions.append(eigenfunction)
            _addends.append(eigenfunction.vector().get_local())

        self.eigenvalues = eigenvalues
        self.eigenfunctions = eigenfunctions

        self.addends = np.vstack(_addends).T

    def plot_eigenvalues(self, outdir):

        import matplotlib.pyplot as plt

        m = self.m
        e = self.eigenvalues
        plt.scatter(range(m), e, s=0.5)
        plt.xlabel("Index of eigenvalue")
        plt.ylabel("Eigenvalue")
        plt.yscale("log")

        plt.tight_layout()
        plt.savefig(outdir + "log_eigenvalues.pdf")
        plt.close()


    def mean_field(self):

        V = self.function_space

        return interpolate(Constant(self.mean), V)

    def sample(self, sample_index):
        return self.realization(sample_index)

    def realization_(self, sample_index):
        """Computes a realization of the random field.

        The seed is increased by one.

        Note:
        -----
        The sum defining the truncated KL expansion is evaluated from smallest
        to largest eigenvalue.
        """
        mu = self.mu
        scale = self.scale
        rf_max = self.random_field_max

        addends = self.addends
        values = self.eigenvalues
        xi = np.sqrt(values) * self.xi_mat[sample_index]
        f = Function(self.function_space)

        y = self.mean_field().vector().get_local()
        y += addends @ xi

        # f.vector()[:] = np.maximum(y, 0.0)
        # f.vector()[:] = [ (scale/rf_max)*smoothed_plus(y[i], mu) for i in range(len(y)) ]
        f.vector()[:] = scale*np.exp(y/rf_max)
        # f.vector()[:] = scale*np.abs(y)/rf_max

        return f

    def realization(self, sample_index):
        """Computes a realization of the random field.

        The seed is increased by one.

        Note:
        -----
        The sum defining the truncated KL expansion is evaluated from smallest
        to largest eigenvalue.
        """
        mu = self.mu
        scale = self.scale
        rf_max = self.random_field_max

        xi = self.xi_mat[sample_index]

        y = self.mean_field().vector().get_local()

        for i in np.argsort(self.eigenvalues):

            value = self.eigenvalues[i]
            fun = self.eigenfunctions[i]

            y += np.sqrt(value)*xi[i]*fun.vector().get_local()

        f = Function(self.function_space)
        # f.vector()[:] = np.maximum(y, 0.0)
        # f.vector()[:] = [ (scale/rf_max)*smoothed_plus(y[i], mu) for i in range(len(y)) ]
        f.vector()[:] = scale*np.exp(y/rf_max)
        # f.vector()[:] = scale*np.abs(y)/rf_max

        return f
