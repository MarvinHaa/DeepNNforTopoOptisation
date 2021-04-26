from fenics import *
import numpy as np

def nu_phi_T(phi, mesh):
    n = FacetNormal(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(DG0)

    h = CellDiameter(mesh)

    nu = assemble((avg(h) * avg(inner(dot(grad(phi), n), dot(grad(phi), n))) * avg(v)) * dS(mesh))

    return np.abs(nu) ** 2


def nu_u_T(u, phi, mesh):
    n = FacetNormal(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(DG0)

    h = CellDiameter(mesh)

    nu = assemble((avg(h) * avg(inner(dot(grad(u) * phi, n), dot(grad(u) * phi, n))) * avg(v)) * dS(mesh))

    return np.abs(nu)


def bulk_criterion(indicator, mesh=None, theta=0.6):
    if mesh is None:
        mesh = indicator.mesh()

    estimate = np.sum([i for i in indicator])
    cell_marker = MeshFunction("bool", mesh, mesh.topology().dim())
    indicator_index = zip(range(len(indicator)), indicator)
    indicator_sorted = sorted(indicator_index, key=lambda ind: ind[1], reverse=True)
    indices, values = zip(*indicator_sorted)
    indcator_cumsum = np.cumsum(values)

    markerArray = np.zeros(len(cell_marker.array()), dtype='bool')
    try:
        indicator_min = next(iter([val for val in indcator_cumsum if val >= theta * estimate]))
        index_sorted_min = np.argmax(indcator_cumsum == indicator_min)
        index_to_mark = np.array(indices[:index_sorted_min + 1])
        markerArray[index_to_mark] = True
    except:
        markerArray = True

    cell_marker.array()[:] = markerArray
    return cell_marker

def interpolateLG(v, space):
    ret = Function(space)
    LagrangeInterpolator.interpolate(ret, v)
    return ret

class meshadaptor():

    def __init__(self, mesh_0, MinIter=50, control=1, maxvertices=15000, verticesMultiplicator =1.5 , theta=0.6, CVaR=False):
        """
        Finite elemente discretization such that the interface region on the pase field phi and the solution of the state equation u are well represented


        :param mesh_0: start mesh from Lineat Elasticity Problem (LEP) (we start refining the new mesh from mesh_0 every time)
        :param MinIter: minimum iteration before refinement
        :param control: value who has to fall below that of the convergence criterion (change(phi)/tau)
        :param maxvertices: maximum vertices of the new mesh
        :param verticesMultiplicator: multiplier for the maximal vertices number for the new mesh
        :param theta: refinement fraction paramerter we use in the bulk criterion
        """
        self.MinIter = MinIter
        self.control = control
        self.iStep = 0
        self.maxvertices = maxvertices
        self.verticesMultiplicator = verticesMultiplicator
        self.mesh_0 = mesh_0
        self.theta = theta
        self.CVaR = CVaR

    def adaptMesh(self, phi_next, u_n, mesh, targetvertices):
        """
        refine mesh_0 until targetvertices are reached

        :param phi_next:
        :param u_n:
        :param mesh:
        :param targetvertices:
        :return: refined mesh
        """
        while mesh.num_vertices() <= targetvertices:
            print("refine mesh..")

            V = FunctionSpace(mesh, VectorElement("CG", mesh.ufl_cell(), 2))
            #F = FunctionSpace(mesh, MixedElement([FiniteElement(t, mesh.ufl_cell(), d) for t, d in [("CG", 1), ("R", 0)]]))
            if self.CVaR:
                F = FunctionSpace(mesh, MixedElement([FiniteElement(t, mesh.ufl_cell(), d) for t, d in [("CG", 1), ("R", 0), ("R", 0)]]))
            else:
                F = FunctionSpace(mesh, MixedElement([FiniteElement(t, mesh.ufl_cell(), d) for t, d in [("CG", 1), ("R", 0)]]))

            phi_next_ = interpolateLG(phi_next, F)
            u_n_ = interpolateLG(u_n, V)

            nu_phi = nu_phi_T(phi_next_[0], mesh)
            nu_u = nu_u_T(u_n_, phi_next_[0], mesh)

            nu_T = np.sqrt(nu_phi) / np.sqrt(sum(nu_phi)) + np.sqrt(nu_u) / np.sqrt(sum(nu_u))

            marked = bulk_criterion(nu_T, mesh, self.theta)

            mesh = refine(mesh, marked)

        return mesh

    def update(self, phi_next,u_n, controler, mesh):
        """
        checks if convergence criterion is reached an start mesh adaption if it's so

        :param phi_next:
        :param u_n:
        :param controler:
        :param mesh:
        :return: indicator if a new mesh is given, new mesh or None
        """
        NewMeshIndicator = False
        new_mesh = None
        self.iStep += 1
        if (controler < self.control and self.iStep > self.MinIter):
            if mesh.num_vertices() < self.maxvertices:
                targetvertices = mesh.num_vertices() * self.verticesMultiplicator
                new_mesh = self.adaptMesh(phi_next, u_n, self.mesh_0, targetvertices)
                NewMeshIndicator = True
                self.iStep = 0
            else:
                new_mesh = mesh
                print("Maximal resolution reached:", mesh.num_vertices())
                self.iStep = 0

        return NewMeshIndicator, new_mesh




