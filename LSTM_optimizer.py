from fenics import *
from Linear_Elasticity_solver_stochastic import *
from adaptivity.tau_adaptivity import tauadaptor
from adaptivity.mesh_adaptivity import meshadaptor

import torch

from Neural_Networks.LSTM.LSTM_classes import EncoderDecoderPhiLSTM
from torch.nn.utils.rnn import pad_sequence

from problems.bridge2D import *

from problems.bridge2Dstochastic import *

import time as tm
import matplotlib.pyplot as plt




global targetmesh
targetmesh = RectangleMesh(Point(-1.0, 0.0), Point(1.0, 1.0), 200, 100, "right/left")


# define global variable
mc_problem = None

LEP = LEProblem(bridge2D())

if LEP.stochastic and LEP.CVaR:
    tau_adapter = tauadaptor(tau_0=LEP.get_tau_stoch(), TOL=1e-1, TOL_t=50, tau_max=1e-5, tau_min=1e-12, tau_t_max=1e1, tau_t_min=1e-12)
else:
    tau_adapter = tauadaptor(LEP.get_tau())

mesh_adapter = meshadaptor(mesh_0=LEP.mesh, CVaR=LEP.CVaR)


def tic():
    global _start_time
    _start_time = tm.time()

def tac():
    t_sec = round(tm.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))




def LSTM_gradienten_step(LEP, phi_val,ux_val,uy_val,model):
    """
    calculates a gradient step with a CNN

    :param LEP: Takes an Linear Elasticity Problem (LEP)
    :param model: given CNN
    :param targetmesh: fenics mesh corresponding to the given CNN in point of dimensions

    """
    original_mesh = LEP.mesh # backup current mesh
    LEP.updateMesh(targetmesh) # project current solution on reference mesh

    phi_coor = LEP.F.tabulate_dof_coordinates()[:-1]

    sequence = []
    # fill matrices with values for suitable coordinates
    for j in range(5):

        # create matrix
        PHI = np.zeros((101, 201))
        Ux = np.zeros((101, 201))
        Uy = np.zeros((101, 201))
        for i in range(len(phi_coor)):
            # project fenics cordinates on matrix coordinates
            x = round((phi_coor[i][0] + 1) * 100)
            y = round((100 - phi_coor[i][1] * 100))
            # filling matrices
            PHI[y, x] = phi_val[j][i]
            Ux[y, x] = ux_val[j][i]
            Uy[y, x] = uy_val[j][i]

        # converting matixes to tensors
        phi_tensor = torch.tensor(PHI)
        ux_tensor = torch.tensor(Ux)
        uy_tenso = torch.tensor(Uy)

        tensor_input = pad_sequence([phi_tensor, ux_tensor, uy_tenso], batch_first=True)

        sequence.append(tensor_input)


    sequence_tensor = pad_sequence(sequence, batch_first=True)

    # predict phi with CNN and transform tensor to numpy matrix
    phi_NN = model(sequence_tensor.float().unsqueeze(0), 10).detach().numpy()[9]


    phi_val = phi_val[-1]
    # convert phi (numpy matrx) to fenics array
    for i in range(len(phi_coor)):
        x = round((phi_coor[i][0] + 1) * 100)
        y = round((100 - phi_coor[i][1] * 100))
        phi_val[i] = phi_NN[y, x]


    LEP.phi_n.vector()[:] = phi_val

    LEP.updateMesh(original_mesh) # project solution back to mehs within the fenics optimization


def deterministic_LSTM_solve(LEP, phi_val,ux_val,uy_val,model):

    """

    Deterministic solvemethod for an Linear Elasticity Problem

    :param LEP: Takes an Linear Elasticity Problem (LEP)
    :return: Solve the corresponding state, adjoint and gradient equation of the given LEP
    """
    # solve state equation
    SE = LEP.SE(LEP.u, LEP.phi_n[0], LEP.v_u, LEP.g)
    solve(lhs(SE) == rhs(SE), LEP.u_n, bcs=LEP.bcSE, solver_parameters={"linear_solver": "umfpack", "preconditioner": "default"}, form_compiler_parameters=None)

    # solve adjoint equation
    LEP.p_n.assign(LEP.u_n)  # p_n = u_n

    # solve gradient equation
    LSTM_gradienten_step(LEP, phi_val,ux_val,uy_val,model)

    LEP.project_phi()

    J_eps = LEP.J_eps(LEP.u_n, LEP.phi_n[0])

    return J_eps


def deterministic_solve(LEP):
    """

    Deterministic solvemethod for an Linear Elasticity Problem

    :param LEP: Takes an Linear Elasticity Problem (LEP)
    :return: Solve the corresponding state, adjoint and gradient equation of the given LEP
    """
    # solve state equation
    SE = LEP.SE(LEP.u, LEP.phi_n[0], LEP.v_u, LEP.g)
    solve(lhs(SE) == rhs(SE), LEP.u_n, bcs=LEP.bcSE, solver_parameters={"linear_solver": "umfpack", "preconditioner": "default"}, form_compiler_parameters=None)

    # solve adjoint equation
    LEP.p_n.assign(LEP.u_n)  # p_n = u_n

    # solve gradient equation
    GE = LEP.GE(LEP.phi, LEP.phi_n, LEP.v_phi, LEP.p_n, LEP.u_n, LEP.tau[0], LEP.gamma)
    solve(lhs(GE) == rhs(GE), LEP.phi_next, bcs=None, solver_parameters={"linear_solver": "umfpack", "preconditioner": "default"}, form_compiler_parameters=None)

    LEP.project_phi()

    print("Phi_n as Vector:", LEP.phi_next.vector()[:])
    print("Länge von Phi:", len(LEP.phi_next.vector()[:]))

    J_eps = LEP.J_eps(LEP.u_n, LEP.phi_n[0])

    return J_eps

def compliance(LEP):
    return assemble(dot(LEP.g, LEP.u_n) * LEP.ds(2))

def get_random_g(LEP, iter=None):

    expr = Expression( ( "1.0" + " * (cos( theta ) * meanX - sin( theta ) *meanY)",
                         "1.0" + " * (sin( theta ) * meanX + cos( theta ) * meanY)" ),
                         meanX = 0, meanY = -5000, theta = 0.0, degree = 1 )

    expr.theta = stats.truncnorm((-pi/2 )/0.3, (pi/2 )/ 0.3, loc=0, scale=0.3 ).rvs(size=1,random_state=iter)[0]

    LEP.g = expr
    return LEP



def run_LSTM_optimization(model_path='Neural_Networks/LSTM/model/ConvLSTM-model_3_9_9_9_9_1_sequence_5in_10out_3618Samples_77batch_100epochs.pth',random_state=0):
    converge = False
    LEP = LEProblem(bridge2Dstochastic())
    if random_state > 0:
        get_random_g(LEP, random_state)
    tau_adapter = tauadaptor(LEP.get_tau())
    mesh_adapter = meshadaptor(mesh_0=LEP.mesh, CVaR=LEP.CVaR)
    model = EncoderDecoderPhiLSTM(nf=9, in_chan=3)

    model.load_state_dict(torch.load(model_path))

    IterationStep = 0
    get_new_tensor_iter = -75
    plot_count = 0
    iters = []
    conv_min_iter = 50

    taus = []
    taus.append(LEP.tau_0)
    gammas = []
    es = []
    controls = []
    Js = []
    mesh_count = 1

    tic()

    k = 0
    times = []
    V = FunctionSpace(targetmesh, VectorElement("CG", targetmesh.ufl_cell(), 1))
    F = FunctionSpace(targetmesh, MixedElement([FiniteElement(t, targetmesh.ufl_cell(), d) for t, d in [("CG", 1), ("R", 0)]]))

    phi_val = []
    ux_val = []
    uy_val = []
    while not converge:
        StartTime = tm.time()
        IterationStep += 1
        get_new_tensor_iter += 1
        plot_count += 1
        iters.append(IterationStep)
        print("\nIter: {IterationStep}".format(IterationStep=IterationStep))

        # solving
        SE = LEP.SE(LEP.u, LEP.phi_n[0], LEP.v_u, LEP.g)
        solve(lhs(SE) == rhs(SE), LEP.u_n, bcs=LEP.bcSE, solver_parameters={"linear_solver": "umfpack", "preconditioner": "default"}, form_compiler_parameters=None)

        if get_new_tensor_iter >= 50 and mesh_count < 2:
            u_n = interpolateLG(LEP.u_n, V)
            phi_n = interpolateLG(LEP.phi_n, F)
            phi_val.append(phi_n.vector()[:])
            ux, uy = u_n.split(deepcopy=True)
            ux_val.append(ux.vector()[:])
            uy_val.append(uy.vector()[:])

            phi_val = phi_val[-5:]
            ux_val = ux_val[-5:]
            uy_val = uy_val[-5:]

        # get u_k and phi_{k-1}
        if get_new_tensor_iter >= 55 and mesh_count < 2:
            deterministic_LSTM_solve(LEP, phi_val, ux_val, uy_val,model)

            get_new_tensor_iter = 0

        # solve adjoint equation

        LEP.p_n.assign(LEP.u_n)  # p_n = u_n

        # solve gradient equation
        GE = LEP.GE(LEP.phi, LEP.phi_n, LEP.v_phi, LEP.p_n, LEP.u_n, LEP.tau[0], LEP.gamma)
        solve(lhs(GE) == rhs(GE), LEP.phi_next, bcs=None, solver_parameters={"linear_solver": "umfpack", "preconditioner": "default"}, form_compiler_parameters=None)

        LEP.project_phi()

        # optimization adaptions
        tau_n = LEP.get_tau()
        taus.append(tau_n)

        e_n = LEP.get_e_n()
        es.append(e_n)

        control = LEP.get_control_change()
        controls.append(control / tau_n)

        gammas.append(LEP.gamma(0))

        J = assemble(LEP.J(LEP.u_n, LEP.phi_next[0], LEP.gamma))

        Js.append(J)

        LEP.tau.vector()[:] = tau_adapter.nextTau(e_n)[0]

        LEP.new_adaptGamma()

        if plot_count >= 100:
            plot(LEP.phi_n[0])
            plt.show()
            plot_count = 0

        NewMeshIndicator, new_mesh = mesh_adapter.update(LEP.phi_next, LEP.u_n, controls[-1], LEP.mesh)
        conv_min_iter += 1
        if NewMeshIndicator:
            mesh_count += 1
            conv_min_iter = 0
            LEP.updateMesh(new_mesh)
            control = 1000
        LEP.phi_last.assign(LEP.phi_n)
        LEP.phi_n.assign(LEP.phi_next)
        k += 1

        if control < 1 and conv_min_iter >= 50 and mesh_count >= 4:  # Kovergenz
            converge = True
            print('convert after:' + str(k) + 'steps')
        if k == 700:
            converge = True
            print('reach may itteration steps')
        EndTime = tm.time()
        times.append(EndTime - StartTime)
    tac()



if __name__ == '__main__':
    run_LSTM_optimization()