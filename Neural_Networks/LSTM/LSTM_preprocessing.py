from optimizer import *
import scipy.stats as stats
import torch
from torch.nn.utils.rnn import pad_sequence

global targetmesh
targetmesh = RectangleMesh(Point(-1.0, 0.0), Point(1.0, 1.0), 200, 100, "right/left")


def get_unique_dist_g(LEP, iter=None):
    expr = Expression( ( "1.0" + " * (cos( theta ) * meanX - sin( theta ) *meanY)",
                         "1.0" + " * (sin( theta ) * meanX + cos( theta ) * meanY)" ),
                         meanX = 0, meanY = -5000, theta = 0.0, degree = 1 )

    expr.theta = stats.truncnorm((-pi/2 )/0.3, (pi/2 )/ 0.3, loc=0, scale=0.3 ).rvs(size=1)[0]

    LEP.g = expr
    return LEP

def sample_opti_sequenz(iter=None, maxIteration = 501):
    # initialize LEP
    LEP = LEProblem(bridge2Dstochastic())
    tau_adapter = tauadaptor(LEP.get_tau())
    mesh_adapter = meshadaptor(mesh_0=LEP.mesh, CVaR=LEP.CVaR)

    LEP = get_unique_dist_g(LEP, iter=None)
    #
    IterationStep = 0
    get_new_tensor_iter=24


    iters = []
    taus = []
    taus.append(LEP.tau_0)
    gammas = []
    es = []
    controls = []
    sequence = []

    for k in range(maxIteration):
        IterationStep += 1
        get_new_tensor_iter +=1
        iters.append(IterationStep)
        print("\nIter: {IterationStep}".format(IterationStep=IterationStep))

        # solving
        SE = LEP.SE(LEP.u, LEP.phi_n[0], LEP.v_u, LEP.g)
        solve(lhs(SE) == rhs(SE), LEP.u_n, bcs=LEP.bcSE, solver_parameters={"linear_solver": "umfpack", "preconditioner": "default"}, form_compiler_parameters=None)

        #get u_k and phi_{k-1}
        if get_new_tensor_iter >= 5:
            original_mesh = LEP.mesh    # secure current mesh
            LEP.updateMesh(targetmesh)  # project current solution on reference mesh
            phi_val = LEP.phi_n.vector()[:-1]   # get arrays
            phi_coor = LEP.F.tabulate_dof_coordinates()[:-1]

            ux, uy = LEP.u_n.split(deepcopy=True)
            ux_val = ux.vector()[:]
            uy_val = uy.vector()[:]

            PHI = np.zeros((101, 201))  # create matrix
            Ux = np.zeros((101, 201))
            Uy = np.zeros((101, 201))

            for i in range(len(phi_coor)):  # fill matrices with values for suitable coordinates
                x = round((phi_coor[i][0] + 1) * 100)   # project fenics cordinates auf matrix coordinates
                y = round((100 - phi_coor[i][1] * 100))
                PHI[y, x] = phi_val[i]  # filling matrices
                Ux[y, x] = ux_val[i]
                Uy[y, x] = uy_val[i]

            #converting matixes to tensorsequenz
            phi_tensor = torch.tensor(PHI)
            ux_tensor = torch.tensor(Ux)
            uy_tenso = torch.tensor(Uy)

            tensor_input = pad_sequence([phi_tensor,ux_tensor,uy_tenso], batch_first=True)

            sequence.append(tensor_input)

            LEP.updateMesh(original_mesh)

            get_new_tensor_iter =0

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

        LEP.tau.vector()[:] = tau_adapter.nextTau(e_n)[0]

        LEP.new_adaptGamma()

        NewMeshIndicator, new_mesh = mesh_adapter.update(LEP.phi_next, LEP.u_n, controls[-1], LEP.mesh)

        if NewMeshIndicator:
            LEP.updateMesh(new_mesh)

        LEP.phi_last.assign(LEP.phi_n)
        LEP.phi_n.assign(LEP.phi_next)

    sequence_tensor = pad_sequence(sequence, batch_first=True)


    torch.save(sequence_tensor, 'data/' + str(iter) + '_(' + str(LEP.g(0)[0]) + ',' + str(LEP.g(0)[1]) + ')_truncnorm_bridge2D_GRAPH_100_sequence.pt')


    return sequence_tensor, LEP



if __name__=='__main__':

    for i in range(2):
        sequence_tensor, LEP = sample_opti_sequenz(maxIteration=501,iter=i)
