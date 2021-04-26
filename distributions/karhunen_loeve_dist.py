from fenics import *
from mshr import *

import numpy as np
import scipy.stats as stats
from scipy.sparse.linalg.eigen import eigsh


class KLdist:

    def __init__(self, mesh, distType='lognormal', mean=0, var=1, nModes=10, cov_lenght=0.1, cov_scal=100, kappaMean=150, kappaScale=1):
        self.mesh = mesh
        self.distType = distType
        if self.distType in ['normal', 'lognormal']:
            self.dist = stats.norm
        else:
            error("unknown Lame-Coefficient distribution")
        self.space = FunctionSpace(self.mesh, "CG", 1)
        self.nModes = nModes
        self.cov_lenght = cov_lenght
        self.cov_scal = cov_scal
        self.eigval, self.eigvec = self.get_eigenValsVecs()
        self.mean = mean
        self.var = var

        self.kappaMean = interpolate(Constant(kappaMean), self.space)
        self.kappaScale = kappaScale

        self.randvec = None

    def get_eigenValsVecs(self):

        nDOF = self.space.dim()  # number of dof
        d = self.mesh.geometry().dim()
        c4dof = self.space.tabulate_dof_coordinates().reshape(nDOF, d)

        u = TrialFunction(self.space)
        v = TestFunction(self.space)

        MassM = assemble(u * v * dx)
        M = MassM.array()

        c0 = np.repeat(c4dof, nDOF, axis=0)
        c1 = np.tile(c4dof, [nDOF, 1])
        r = np.linalg.norm(c0 - c1, axis=1)
        C = self.cov_scal * np.exp(-r ** 2 / self.cov_lenght ** 2)  # covariance
        C.shape = [nDOF, nDOF]

        A = np.dot(M, np.dot(C, M))
        w, v = eigsh(A, self.nModes, M)

        dof2vert = self.space.dofmap().dofs()
        eigenVectors = np.array([z[dof2vert] for z in v.T])

        eigenVectors = list(eigenVectors)
        eigenValues = list(w)

        return eigenValues[::-1], eigenVectors[::-1]

    def sample(self, random_state=None):
        if random_state is not None:
            np.random.seed(seed=random_state)

        self.randVec = self.dist.rvs(loc=self.mean, scale=self.var, size=self.nModes)

        sampVec = sum(np.array([val * vec * rand for val, vec, rand in zip(self.eigval, self.eigvec, self.randVec)]))

        if self.distType.startswith("log"):
            sampVec = np.exp(sampVec)

        sampVec = sampVec * self.kappaScale + self.kappaMean.vector().get_local()

        self.expression = Function(self.space)
        self.expression.vector()[:] = sampVec

        return self.expression


if __name__ == '__main__':
    resolution = 30
    upperEnd = 1.0
    lowerEnd = 0.0
    leftEnd = -1.0
    rightEnd = 1.0
    dirichletSize = 0.1
    neumannSize = 0.02

    domain = (Rectangle(Point(leftEnd, lowerEnd), Point(leftEnd + dirichletSize, upperEnd))
              + Rectangle(Point(leftEnd + dirichletSize, lowerEnd), Point(-neumannSize, upperEnd))
              + Rectangle(Point(-neumannSize, lowerEnd), Point(neumannSize, upperEnd))
              + Rectangle(Point(neumannSize, lowerEnd), Point(rightEnd - dirichletSize, upperEnd))
              + Rectangle(Point(rightEnd - dirichletSize, lowerEnd), Point(rightEnd, upperEnd))
              )

    mesh = generate_mesh(domain, resolution)

    lame_field = KLdist(mesh, nModes=10, cov_scal=1, kappaScale=100)
    import matplotlib.pyplot as plt

    plt.figure()
    plot(lame_field.sample())
    plt.show()
