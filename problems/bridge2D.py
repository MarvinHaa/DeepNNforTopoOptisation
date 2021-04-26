from fenics import *
from mshr import *


def bridge2D():
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

    #mesh = RectangleMesh(-1.0, 0.0, 1.0, 1.0, 10, 10, "right/left")

    parameter = {"mesh": mesh,
                 "g": Constant((0, -5000)),
                 "phi_0": Constant(0.5),
                 "gammaD": GammaD(),
                 "gammaG": GammaG(),
                 "gammaS": GammaS(),
                 "SDir": 1,
                 "lmbda": Constant(150),
                 "mu": Constant(150),
                 "m": 0.4,
                 "dim": 2,
                 "gamma": Constant(1.0),
                 "epsilon": Constant(1. / 16),
                 "CVaR": False,
                 "stochastic": False
                 }
    return parameter
