from fenics import *

import numpy as np
import scipy.stats as stats
from functools import partial


class rotationdist:

    def __init__(self, distType='normal_trunc', scale=0.3, mean=0, kappaMean=(0,-5000), limits=[-pi/2, pi/2], axis='yz'):
        self.kappaMean = kappaMean
        self.lowbound = limits[0]
        self.upbound = limits[1]
        self.dim = len(kappaMean)
        self.axis = axis

        if distType == 'normal':
            self.dist = partial( stats.norm.rvs, loc = mean, scale = scale )
        elif distType == 'uniform':
            self.dist = stats.uniform(loc=-1, scale=2).rvs
        elif distType == 'normal_trunc':
            self.dist = stats.truncnorm((self.lowbound - mean)/scale, (self.upbound - mean)/ scale, loc=mean, scale=scale ).rvs
            #self.distQMC = partial(quasiStdTruncNormal, upperLimit=self.upbound, lowerLimit=self.lowbound, mu=mean, sigma=scale)
        else:
            error("unknown rotation distribution")

        self._setExpression()

    def _setExpression( self ):
        if self.dim == 2:
            self.expr = Expression( ( "1.0" + " * (cos( theta ) * meanX - sin( theta ) *meanY)",
                         "1.0" + " * (sin( theta ) * meanX + cos( theta ) * meanY)" ),
                         meanX = self.kappaMean[0], meanY = self.kappaMean[1], theta = 0.0, degree = 1 )
        elif self.dim == 3:
            if self.axis == "yz":
                self.expr = Expression(
                    ( "1.0" + " * (cos(theta2) * cos(theta1) * meanX - sin(theta2) * meanY + cos(theta2) * sin(theta1) * meanZ)",
                      "1.0" + " * (sin(theta2) * cos(theta1) * meanX + cos(theta2) * meanY + sin(theta2) * sin(theta1) * meanZ)",
                      "1.0" + " * (-sin(theta1) * meanX + cos(theta1) * meanZ)" ),
                    theta1 = 0.0, theta2 = 0.0,
                    meanX = self.kappaMean[0], meanY = self.kappaMean[1], meanZ = self.kappaMean[2],
                    degree = 1
                    )
            elif self.axis == "zy":
                self.expr = Expression(
                    ( "1.0" + " * (cos(theta2) * cos(theta1) * meanX - cos(theta2) * sin(theta1) * meanY + sin(theta2) * meanZ)",
                      "1.0" + " * (sin(theta1) * meanX + cos(theta1) * meanY)",
                      "1.0" + " * (-sin(theta2) * cos(theta1) * meanX -  sin(theta2)*sin(theta1)* meanY + cos(theta2) * meanZ)" ),
                    theta1 = 0.0, theta2 = 0.0,
                    meanX = self.kappaMean[0], meanY = self.kappaMean[1], meanZ = self.kappaMean[2],
                    degree = 1
                    )
            else:
                error( "Unknown axis combination" )
        else:
            error( "Unknown number of dimensions" )


    def sample(self,random_state=None):
        if random_state is not None:
            np.random.seed(seed=random_state)
        if self.dim ==2:
            self.expr.theta = self.dist(size=1)[0]
        else:
            self.expr.theta1 = self.dist(size=1)[0]
            self.expr.theta2 = self.dist2(size=1)[0]
        return self.expr

if __name__ == '__main__':

    rd = rotationdist()

    import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plot(rd.sample())
    # plt.show()
    origin = [0], [0]
    a, b = rd.sample()(0)
    #b = rd.sample()(0)[1]
    print(a,b)
    plt.quiver(*origin, a, b)
    plt.show()