from fenics import *
from mshr import *


def mast2D():
    scale = 120.

    height = 120 / scale
    width = 80 / scale
    footHeight = 80 / scale
    footWidth = 40 / scale
    loadSize = 5 / scale
    standSize = 5 / scale

    headLowerSize = (width - footWidth) / 2.0
    nonLoadSize = headLowerSize - loadSize
    headHeight = height - footHeight

    domain = (Rectangle(Point(0.0, 0.0), Point(width, height))
              - Rectangle(Point(0.0, 0.0), Point(headLowerSize, footHeight))
              - Rectangle(Point(headLowerSize + footWidth, 0.0), Point(width, footHeight))
              )
    resolution = 40

    class GammaD(SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and near(x[1], 0)
                    and (x[0] <= (headLowerSize + standSize) or x[0] >= (headLowerSize + footWidth - standSize))
                    )

    class GammaG(SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and near(x[1], footHeight)
                    and ( x[0] <= loadSize or x[0] >= ( width - loadSize ) )
                     )

    class GammaS(SubDomain):
        def inside(self, x, on_boundary):
            return False  # cantilever problem has no slip condition

    mesh = generate_mesh(domain, resolution)

    parameter = {"mesh": mesh,
                 "g": Constant((0, -200)),
                 "phi_0": Constant(0.5),
                 "gammaD": GammaD(),
                 "gammaG": GammaG(),
                 "gammaS": GammaS(),
                 "SDir": 1,
                 "lmbda": Constant(250),
                 "mu": Constant(250),
                 "m": 0.3,
                 "dim": 2,
                 "gamma": Constant(4.0),
                 "epsilon": Constant(1. / 16)
                 }
    return parameter