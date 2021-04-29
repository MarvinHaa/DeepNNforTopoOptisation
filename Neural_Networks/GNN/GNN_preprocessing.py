from optimizer import *
import scipy.stats as stats
import torch

import dgl
import itertools

def get_unique_dist_g(LEP, iter=None):
    expr = Expression(("1.0" + " * (cos( theta ) * meanX - sin( theta ) *meanY)",
                       "1.0" + " * (sin( theta ) * meanX + cos( theta ) * meanY)"),
                      meanX=0, meanY=-5000, theta=0.0, degree=1)

    expr.theta = stats.truncnorm((-pi / 2) / 0.3, (pi / 2) / 0.3, loc=0, scale=0.3).rvs(size=1, random_state=iter)[0]

    LEP.g = expr
    return LEP

def sample_opti_sequenz_graph(iter=None, maxIteration = 501):
    # initialize LEP
    LEP = LEProblem(bridge2Dstochastic())
    tau_adapter = tauadaptor(LEP.get_tau())
    mesh_adapter = meshadaptor(mesh_0=LEP.mesh, CVaR=LEP.CVaR)

    LEP = get_unique_dist_g(LEP, iter=None)

    IterationStep = 0
    get_new_graph_iter = 4

    iters = []
    taus = []
    taus.append(LEP.tau_0)
    gammas = []
    es = []
    controls = []

    for k in range(maxIteration):
        IterationStep += 1
        get_new_graph_iter +=1
        iters.append(IterationStep)
        print("\nIter: {IterationStep}".format(IterationStep=IterationStep))

        # solving
        SE = LEP.SE(LEP.u, LEP.phi_n[0], LEP.v_u, LEP.g)
        solve(lhs(SE) == rhs(SE), LEP.u_n, bcs=LEP.bcSE, solver_parameters={"linear_solver": "umfpack", "preconditioner": "default"}, form_compiler_parameters=None)

        #get u_k and phi_{k-1}
        if get_new_graph_iter >= 5:

            # get edges from fenics
            edge_from = []
            edge_to = []
            #element = LEP.F.element()
            dofmap = LEP.F.dofmap()
            for cell in cells(LEP.mesh):
                finite_element_node_index = dofmap.cell_dofs(cell.index())
                edge_from += list(list(zip(*itertools.permutations(finite_element_node_index[:-1], 2)))[0])
                edge_to += list(list(zip(*itertools.permutations(finite_element_node_index[:-1], 2)))[1])

            # initialize graph
            g = dgl.graph((edge_from, edge_to)) # initialize Graph (nodes and edges)


            g.ndata['value'] = torch.tensor([ [0.5] for i in range(g.number_of_nodes())], dtype=torch.float32)
            g.ndata['position'] = torch.tensor([ [0.001, 0.001] for i in range(g.number_of_nodes()) ], dtype=torch.float32)
            g.ndata['u'] = torch.tensor([ [0.001, 0.001] for i in range(g.number_of_nodes()) ], dtype=torch.float32)


            # get graph data from fenics
            phi_val = LEP.phi_n.vector()[:-1]
            ux, uy = LEP.u_n.split(deepcopy=True)
            ux_val = ux.vector()[:]
            uy_val = uy.vector()[:]
            phi_coor = LEP.F.tabulate_dof_coordinates()[:-1]


            for i in range(len(phi_val)):
                g.ndata['value'][i] = phi_val[i]
                g.ndata['u'][i] = torch.tensor([ux_val[i], uy_val[i]], dtype=torch.float32)
                g.ndata['position'][i] = torch.tensor(phi_coor[i], dtype=torch.float32)

            dgl.save_graphs('data/' + str(iter)+ '_' +str(IterationStep)+ '_(' + str(LEP.g(0)[0]) + ',' + str(LEP.g(0)[1]) + ')_truncnorm_bridge2D_GRAPH_50_sequence.bin', g)


            get_new_graph_iter = 0


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



    return



if __name__=='__main__':

    for i in range(2):
        sample_opti_sequenz_graph(maxIteration=500,iter=i)
