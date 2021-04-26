from fenics import *
#from problems.bridge2D import *
from problems.bridge2Dstochastic import *

#import matplotlib.pyplot as plt
from adaptivity.tau_adaptivity import tauadaptor
from adaptivity.mesh_adaptivity import meshadaptor
import numpy as np

# support functions__________________________________________
def interpolateLG(v, space):
    ret = Function(space)
    LagrangeInterpolator.interpolate(ret, v)
    return ret


class LEProblem():

    def __init__(self, problemparameter):
        # Load problem parameter's---------------------------------------------
        self.problem = problemparameter

        self.solvecount = 0
        self.mesh = self.problem['mesh']
        self.g = self.problem['g']
        self.phi_0 = self.problem['phi_0']
        self.GammaD = self.problem['gammaD']
        self.GammaG = self.problem['gammaG']
        self.GammaS = self.problem['gammaS']
        self.SDir = self.problem['SDir']
        self.lmbda = self.problem['lmbda']
        self.mu = self.problem['mu']
        self.m = self.problem['m']
        self.dim = self.problem['dim']
        self.gamma = self.problem['gamma']
        self.epsilon = self.problem['epsilon']

        self.stochastic = self.problem['stochastic']
        if self.stochastic:
            self.nSamples = self.problem['nSamples']
            self.RandomVariables = self.problem['RandomVariables']
            self.beta = self.problem['beta']
        self.CVaR = self.problem['CVaR']

        self.quasiMClastSeed = 0

        # set default parameter's----------------------------------------------
        if self.stochastic and self.CVaR:
            self.tau_0 = (Constant(1e-9), Constant(1))
            self.c_j = self.problem['gammaAdaptFactor']
        else:
            self.tau_0 = Constant(1e-9)
            self.c_j = Constant(0.05)


        self.meshes = [self.mesh]
        self.lagrange_0 = {}

        # Define Spaces and functions------------------------------------------
        self.V = FunctionSpace(self.mesh, VectorElement("CG", self.mesh.ufl_cell(), 1))
        if self.stochastic and self.CVaR:
            self.F = FunctionSpace(self.mesh, MixedElement([FiniteElement(t, self.mesh.ufl_cell(), d) for t, d in [("CG", 1), ("R", 0), ("R", 0)]]))
        else:
            self.F = FunctionSpace(self.mesh, MixedElement([FiniteElement(t, self.mesh.ufl_cell(), d) for t, d in [("CG", 1), ("R", 0)]]))


        self.u_n = Function(self.V)
        self.p_n = Function(self.V)
        self.phi_n = Function(self.F)

        self.phi_next = Function(self.F)
        self.phi_last = Function(self.F)
        self.tau = Function(self.F)

        if self.stochastic and self.CVaR:
            for subspace, tau in zip((0,2), self.tau_0): #0=Phi ("CG", 1), 2=t ("R", 0)
                dofs = self.F.sub(subspace).dofmap().dofs()
                self.tau.vector()[dofs] = tau
        else:
            self.tau.vector()[:] = self.tau_0

        self.u = TrialFunction(self.V)
        self.v_u = TestFunction(self.V)
        self.p = TrialFunction(self.V)
        self.v_p = TestFunction(self.V)
        self.phi = TrialFunction(self.F)
        self.v_phi = TestFunction(self.F)

        if self.stochastic:
            self.var_u = Function(self.V)
            self.var_p = Function(self.V)
            self.var_phi = Function(self.F)

            self.damp_phi = Function(self.F)

        self.lagrange_0["0"] = self.phi_0
        if self.stochastic:
            self.lagrange_0["2"] = Constant(0)

        for i in range(0, self.F.num_sub_spaces()):
            val = self.lagrange_0.get(str(i), Constant(1))
            assign(self.phi_next.sub(i), project(val, self.F.sub(i).collapse()))
        self.phi_n.assign(self.phi_next)
        self.phi_last.assign(self.phi_next)

        # Initialize boundary's------------------------------------------------

        class GammaN(SubDomain):

            def inside(self, x, on_boundary):
                return on_boundary

        self.gammaN = GammaN()


        self.boundaries = MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        self.GammaD.mark(self.boundaries, 1)
        self.GammaG.mark(self.boundaries, 2)
        self.GammaS.mark(self.boundaries, 3)

        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)

        self.homogeneousBC = Constant(tuple(0.0 for i in range(self.dim)))
        self.bcSE = [DirichletBC(self.V, self.homogeneousBC, self.boundaries, 1),
                     DirichletBC(self.V.sub(self.SDir), Constant(0.0), self.boundaries, 3)]

        self.bcGE = [DirichletBC(self.F.sub(0), Constant(1.0), self.boundaries, 1),  # GammaD
                     DirichletBC(self.F.sub(0), Constant(1.0), self.boundaries, 2),  # GammaG
                     ]

        # Formulate problem's--------------------------------------------------
        # ε(u) = sym(∇u)
        self.Epsilon = lambda u: 0.5 * (grad(u) + grad(u).T)

        # w(φ) = max(φ^3,0)
        self.omega = lambda phi: conditional(lt(phi, 0), 0, phi ** 3)

        # δw(φ)/δφ  = max(3φ^2,0)
        self.omega_dphi = lambda phi_n: conditional(lt(phi_n, 0), 0, 3 * phi_n ** 2)

        # C(u) = 2µε(u) + λ tr(ε(u)I)
        self.C = lambda u: 2 * self.mu * self.Epsilon(u) + self.lmbda * tr(self.Epsilon(u)) * Identity(self.dim)

        # 𝝈(φ,u) = C(u)w(φ) + eps^2C(u)w(1-φ) - Stress
        self.sigma = lambda phi, u: self.C(u) * self.omega(phi) + Constant(self.epsilon ** 2) * self.C(u) * self.omega(1 - phi)

        # ψ_0(φ) = (φ - φ^2)^2
        self.psi_0 = lambda phi_next: (phi_next - phi_next ** 2) ** 2

        # ∫_D 𝝈(φ,u):ε(v_u) dx = ∫_(Γ_g) g v_u ds
        self.SE = lambda u, phi, v_u, g: inner(self.sigma(phi, u), self.Epsilon(v_u)) * dx(self.mesh) - dot(g, v_u) * self.ds(2)

        # δψ_0(φ)/δφ = 2(φ - φ^2)(1-2φ)
        self.psi_0_dphi = lambda phi_n: 2 * (phi_n - phi_n ** 2) * (1 - 2 * phi_n)

        # δJ/δφ = eps γ ∫_D  ∇φ*∇v_φ dx + γ/eps ∫_D δψ_0(φ)/δφ*v_φ dx
        self.J_dphi = lambda phi, phi_n, v_phi, gamma: self.epsilon * gamma * inner(grad(phi), grad(v_phi)) * dx(self.mesh) + gamma / self.epsilon * dot(self.psi_0_dphi(phi_n), v_phi) * dx(self.mesh)

        # δ𝝈(φ,p)/δφ = C(p)*δw(φ)/δφ - eps^2C(p)*δw(φ)/δφ
        self.sigma_dphi = lambda phi_n, u: self.C(u) * self.omega_dphi(phi_n) - Constant(self.epsilon ** 2) * self.C(u) * self.omega_dphi(phi_n)

        # δS/δφ = -∫_D δ𝝈(φ,p)/δφ: v_φε(u) dx
        self.S_dphi = lambda p, u, phi_n, v_phi: -(inner(self.Epsilon(p), self.sigma_dphi(phi_n, u) * v_phi) * dx(self.mesh))

        # δM/δφ = ∫_D (φ - m) v_λ dx + ∫_D λ v_φ dx
        self.M_dphi = lambda lmda, v_phi, v_lmda, phi: dot(phi - self.m, v_lmda) * dx(self.mesh) - dot(v_phi, lmda) * dx(self.mesh)

        # DJ^eps = δJ/δφ + δS/δφ + δM/δφ
        self.D_J = lambda phi, phi_n, lmda, v_phi, p, u, gamma: self.J_dphi(phi, phi_n, v_phi[0], gamma) + self.S_dphi(p, u, phi_n, v_phi[0]) + self.M_dphi(lmda, v_phi[0], v_phi[1], phi)

        # eps/τ ∫_D (φ - φ_n) v_φ dx + DJ^eps = 0
        self.GE = lambda phi_lmda, phi_n, v_phi, p, u, tau, gamma: self.epsilon / tau * (phi_lmda[0] - phi_n[0]) * v_phi[0] * dx(self.mesh) + self.D_J(phi_lmda[0], phi_n[0], phi_lmda[1], v_phi, p, u,gamma)

        self.r_t = lambda v_phi, u: v_phi[2]* dx(self.mesh) if (assemble( dot(self.g, u) *self.ds(2) ) - self.get_t()) <= 0 else (1 - (1/(1-self.beta)))*v_phi[2]*dx(self.mesh)

        self.GE_CVaR = lambda phi_lmda, phi_n, v_phi, p, u, tau, gamma: self.epsilon / tau[0] * (phi_lmda[0] - phi_n[0]) * v_phi[0] * dx(self.mesh) \
                                                                        + self.epsilon/ tau[2] * (phi_lmda[2] - phi_n[2]) * v_phi[2] *dx(self.mesh) \
                                                                        + self.D_J(phi_lmda[0], phi_n[0], phi_lmda[1], v_phi, p, u, gamma) \
                                                                        + self.r_t(v_phi, u)

        # E^eps(φ) = ∫_D eps/2 |∇φ|^2 + 1/2 ψ_0(φ) dx
        self.E_eps = lambda phi: ((self.epsilon / 2) * dot(grad(phi), grad(phi)) + (1 / 2) * self.psi_0(phi)) * dx(self.mesh)

        # J^eps(φ) = ∫_(Γ_g) g u(φ) ds + γE^eps(φ)
        self.J = lambda u, phi, gamma: dot(self.g, u) * self.ds(2) + gamma * self.E_eps(phi)

        # compliance measure
        self.compliance = lambda u: dot(self.g * u) * self.ds(2)

        self.J_eps = lambda u, phi: assemble(dot(self.g,u) * self.ds(2) + self.gamma * self.E_eps(phi))

    # Define Problem functionalities-------------------------------------------
    def project_phi(self):
        """

        :return: Projection of Phi to [0,1].
        """
        phiDofs = self.F.sub(0).dofmap().dofs()
        self.phi_next.vector()[phiDofs] = np.clip(self.phi_next.vector().get_local()[phiDofs], 0, 1)

    def get_e_n(self):
        """

        :return: Drive relative change of Phi in the L2 norm ||phi_n - phi_(n+1)||_L(D)^2 / ||phi_(n+1)||_L(D)^2
        """
        L2change = sqrt(assemble((self.phi_n[0] - self.phi_next[0]) ** 2 * dx(self.mesh)))
        L2phi_next = sqrt(assemble(self.phi_next[0] ** 2 * dx(self.mesh)))
        return np.array([L2change / L2phi_next])

    def get_e_n_stoch(self):
        """
        Drive relative change of Phi and t in the L2 norm ||(phi_n, t_n) - (phi_(n+1),t_(n+1))||_L(D)^2 / ||(phi_(n+1),t_(n+1))||_L(D)^2

        :return: [e_phi,e_t]
        """
        diffFun = Function(self.phi_n.function_space())
        diffFun.vector()[:] = np.absolute(self.phi_n.vector().get_local() - self.phi_next.vector().get_local())#/np.absolute(self.phi_next.vector().get_local())

        diffFun = project(diffFun, self.F, solver_type="umfpack", preconditioner_type="default")
        return np.array(tuple(subSpaceVector(diffFun, ssid).max() for ssid in (0,2)))

    def get_control_change(self):
        """

        :return:  Drive relative change of Phi in the L2 norm (||phi_n - phi_(n+1)||_L(D)^2)^2 / (||phi_(n+1)||_L(D)^2)^2
        """
        if self.stochastic:
            damp_phi_old = self.damp_phi
            self.damp_phi = Function(self.damp_phi.function_space())
            self.damp_phi.vector()[:] = (0.9 * damp_phi_old.vector().get_local() + (1 - 0.9) * self.phi_next.vector().get_local())
            return assemble((self.damp_phi[0] - damp_phi_old[0]) ** 2 * dx(self.mesh)) / assemble(self.damp_phi[0] ** 2 * dx(self.mesh))
        else:
            return assemble((self.phi_next[0] - self.phi_n[0]) ** 2 * dx) / assemble(self.phi_next[0] ** 2 * dx)

    def new_adaptGamma(self):
        """

        :return: Drive gammma for balancing the twi contributions in the functional with respect to the factor c_j (= 0.1 by default)
        """
        self.gamma = Constant(self.c_j * 1 / assemble(self.E_eps(self.phi_next[0])) * assemble(dot(self.g, self.u_n) * self.ds(2)))

    def get_tau(self):
        """
        Drive tau
        :return: tau
        """
        space = self.tau.function_space()
        subDofs = space.sub(0).dofmap().dofs()
        fun = self.tau.vector().get_local()
        vec = fun[subDofs]

        if len(vec) == 1:
            vec = vec[0]
        return np.array([np.mean(vec)])

    def get_tau_stoch(self):
        """
        Drive tua_phi and tau_t

        :return: [tau_phi, tau_t]
        """
        dofs_tau_phi = self.F.sub(0).dofmap().dofs()
        dofs_tau_t = self.F.sub(2).dofmap().dofs()
        fun = self.tau.vector().get_local()

        tau_phi_vec = fun[dofs_tau_phi]
        tau_t_vec = fun[dofs_tau_t]


        tau_phi_vec = np.mean(tau_phi_vec[0])


        tau_t_vec = np.mean(tau_t_vec)

        return np.array([tau_phi_vec, tau_t_vec])

    def update_tau_stoch(self, new_tau):
        """
        Update tua_phi and tau_t

        :param new_tau: tupel of scalars
        :return:
        """
        dofs_tau_phi = self.F.sub(0).dofmap().dofs()
        self.tau.vector()[dofs_tau_phi] = new_tau[0]
        dofs_tau_t = self.F.sub(2).dofmap().dofs()
        self.tau.vector()[dofs_tau_t] = new_tau[1]

    def updateMesh(self, new_mesh):
        """

        :param new_mesh: new mesh given by the mesh adaptivity
        :return: update mesh, corresponding spaces and interpolate all functions in new spaces
        """
        self.mesh = new_mesh
        self.meshes.append(self.mesh)

        # Define Space's on new mesh
        self.V = FunctionSpace(self.mesh, VectorElement("CG", self.mesh.ufl_cell(), 1))
        #self.F = FunctionSpace(self.mesh, MixedElement([FiniteElement(t, self.mesh.ufl_cell(), d) for t, d in [("CG", 1), ("R", 0)]]))

        if self.stochastic and self.CVaR:
            self.F = FunctionSpace(self.mesh, MixedElement([FiniteElement(t, self.mesh.ufl_cell(), d) for t, d in [("CG", 1), ("R", 0), ("R", 0)]]))
        else:
            self.F = FunctionSpace(self.mesh, MixedElement([FiniteElement(t, self.mesh.ufl_cell(), d) for t, d in [("CG", 1), ("R", 0)]]))

        self.u = TrialFunction(self.V)
        self.v_u = TestFunction(self.V)
        self.p = TrialFunction(self.V)
        self.v_p = TestFunction(self.V)
        self.phi = TrialFunction(self.F)
        self.v_phi = TestFunction(self.F)

        # Interpolate functions in new space's
        self.u_n = interpolateLG(self.u_n, self.V)
        self.p_n = interpolateLG(self.p_n, self.V)
        self.phi_n = interpolateLG(self.phi_n, self.F)
        self.phi_last = interpolateLG(self.phi_last, self.F)
        self.phi_next = interpolateLG(self.phi_next, self.F)
        self.tau = interpolateLG(self.tau, self.F)

        if self.stochastic:
            self.var_u = interpolateLG(self.var_u, self.V)
            self.var_p = interpolateLG(self.var_p, self.V)
            self.var_phi = interpolateLG(self.var_phi, self.F)

            self.damp_phi = interpolateLG(self.damp_phi, self.F)

        # Initialize boundary's on new mesh
        self.boundaries = MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        self.GammaD.mark(self.boundaries, 1)
        self.GammaG.mark(self.boundaries, 2)
        self.GammaS.mark(self.boundaries, 3)

        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)

        self.homogeneousBC = Constant(tuple(0.0 for i in range(self.dim)))
        self.bcSE = [DirichletBC(self.V, self.homogeneousBC, self.boundaries, 1),
                     DirichletBC(self.V.sub(self.SDir), Constant(0.0), self.boundaries, 3)]

        self.bcGE = [DirichletBC(self.F.sub(0), Constant(1.0), self.boundaries, 1),  # GammaD
                     DirichletBC(self.F.sub(0), Constant(1.0), self.boundaries, 2),  # GammaG
                     ]

    def sample(self, random_state=None):
        """
        Sample all randomfunctions
        :param random_state: int
        :return:
        """
        for rv in self.RandomVariables:
            samp = rv[1].sample(random_state)
            for key in rv[0]:
                self.__dict__[key] = samp

    def get_t(self):
        """

        :return: t
        """
        space = self.phi_n.function_space()
        subDofs = space.sub(2).dofmap().dofs()
        fun = self.phi_n.vector().get_local()
        vec = fun[subDofs]
        return vec[0]

def subSpaceVector(fun, subSpaceID):
    """
    Helperfunction to get array of fenics function
    :param fun: fenics function
    :param subSpaceID:
    :return: array
    """
    space = fun.function_space()
    subDofs = space.sub(subSpaceID).dofmap().dofs()
    fun = fun.vector().get_local()
    vec = fun[subDofs]
    if len(vec) == 1:
        vec = vec[0]
    return vec


class ObjectWrapper():
    r"""
    Wraps an object such that new members do not alter the original object but upon access, these new members overload
    members of the original problem. Otherwise this object will behave like the original.
    """

    def __init__(self, obj):
        r"""
        Create the object wrapper.

        Parameters:
            obj (Object): Object to protect.
        """
        self.obj = obj

    def __getattr__(self, attr):  # only called if not in wrapper dict
        r"""
        Get a member of this object. If the member is not found in this object, the wrapped object will be queried.

        Parameters:
            attr (string): Name of the attribute/member to retrieve

        Returns:
            member: attribute or member of the wrapper or the wrapped object by that preference.
        """
        return getattr(self.obj, attr)

if __name__ == '__main__':
    bridge = LEProblem(bridge2Dstochastic())
    tau_adapter = tauadaptor(bridge.get_tau())
    mesh_adapter = meshadaptor(mesh_0=bridge.mesh, CVaR=bridge.CVaR)

    bridge.sample(0)


    for i in range(20):
        monte_carlo_solve(bridge)
        bridge.tau.vector()[:] = tau_adapter.nextTau(bridge.get_e_n())
        bridge.new_adaptGamma()

    plt.figure()
    plot(bridge.mesh)
    plt.show()
    plt.figure()
    plot(bridge.phi_next[0])
    plt.show()
    plt.figure()
    plot(bridge.u_n)
    plt.show()

    new_mesh = mesh_adapter.adaptMesh(bridge.phi_next, bridge.u_n, bridge.mesh, bridge.mesh.num_vertices() * 1.5)

    plt.figure()
    plot(new_mesh)
    plt.show()