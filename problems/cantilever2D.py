from fenics import *
from mshr import *


def cantilever2D():
    resolution = 30
    upperEnd = 1.0
    lowerEnd = 0.0
    leftEnd = -1.0
    rightEnd = 1.0
    neumannSize = 0.25

    class GammaD(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], leftEnd)  # left boundary

    class GammaG(SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and near(x[1], 0)  # lower boundary
                    and rightEnd - neumannSize <= x[0] <= rightEnd)  # middle part

    class GammaS(SubDomain):
        def inside(self, x, on_boundary):
            return False  # cantilever problem has no slip condition

    domain = (Rectangle(Point(leftEnd, lowerEnd), Point(rightEnd - neumannSize, upperEnd))
              + Rectangle(Point(rightEnd - neumannSize, lowerEnd), Point(rightEnd, upperEnd))
              )

    mesh = generate_mesh(domain, resolution)

    parameter = {"mesh": mesh,
                 "g": Constant((0, -600)),
                 "phi_0": Constant(0.5),
                 "gammaD": GammaD(),
                 "gammaG": GammaG(),
                 "gammaS": GammaS(),
                 "SDir": 1,
                 "lmbda": Constant(5000),
                 "mu": Constant(5000),
                 "m": 0.4,
                 "dim": 2,
                 "gamma": Constant(0.1),
                 "epsilon": Constant(1. / 32)
                 }

    return parameter