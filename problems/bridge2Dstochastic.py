from fenics import *
from mshr import *
import scipy.stats as stats
from distributions.karhunen_loeve_dist import *
from distributions.random_rotation import *


def bridge2Dstochastic():
    resolution = 30
    upperEnd = 1.0
    lowerEnd = 0.0
    leftEnd = -1.0
    rightEnd = 1.0
    dirichletSize = 0.1
    neumannSize = 0.02


    class GammaD(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0) and (leftEnd <= x[0] <= (leftEnd + dirichletSize))

    class GammaS(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0) and ((rightEnd - dirichletSize) <= x[0] <= rightEnd)

    class GammaG(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0) and -neumannSize <= x[0] <= neumannSize

    domain = (Rectangle(Point(leftEnd, lowerEnd), Point(leftEnd + dirichletSize, upperEnd))
              + Rectangle(Point(leftEnd + dirichletSize, lowerEnd), Point(-neumannSize, upperEnd))
              + Rectangle(Point(-neumannSize, lowerEnd), Point(neumannSize, upperEnd))
              + Rectangle(Point(neumannSize, lowerEnd), Point(rightEnd - dirichletSize, upperEnd))
              + Rectangle(Point(rightEnd - dirichletSize, lowerEnd), Point(rightEnd, upperEnd))
              )

    mesh = generate_mesh(domain, resolution)

    #mesh = RectangleMesh(Point(-1.0, 0.0), Point(1.0, 1.0), 50, 25, "right/left")

    kl_field = KLdist(mesh, distType='lognormal', nModes=10, cov_lenght=0.1, cov_scal=1, kappaMean=150, kappaScale=100)
    g_field = rotationdist(distType='normal_trunc', scale=0.3, mean=0, kappaMean=(0,-5000))


    parameter = {"mesh": mesh,
                 "stochastic": False,
                 "CVaR": False,
                 "RandomVariables": [(["g"], g_field), (["mu", "lmbda"], kl_field)],
                 "g": Constant((0, -5000)),
                 "phi_0": Constant(0.5),
                 "gammaD": GammaD(),
                 "gammaG": GammaG(),
                 "gammaS": GammaS(),
                 "SDir": 1,
                 "lmbda": Constant(150),
                 "mu": Constant(150),
                 "m": Constant(0.4),
                 "dim": 2,
                 "nSamples": 9,
                 "gamma": Constant(1),
                 "gammaAdaptFactor": Constant(0.01), # 0.1 for Beta > 0
                 "epsilon": Constant(1. / 16),
                 "beta": Constant(0)
                 }
    return parameter