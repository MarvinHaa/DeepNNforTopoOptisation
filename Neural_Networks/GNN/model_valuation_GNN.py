from fenics import *
from Neural_Networks.GNN.GNN_classes import GraphConvPhi

import torch
import torch.nn as nn
from Neural_Networks.GNN.GraphDataset import GraphDataset
from torch.utils.data import DataLoader
import dgl

from optimizer import *
import scipy.stats as stats
import itertools

def _collate_fn(batch):
    g = dgl.batch(batch, edata=None)
    label = g.ndata['labels']
    return g, label

batchsize = 1

model = GraphConvPhi(in_feats=3, hid_feats=9)
model.load_state_dict(torch.load('model/Convelutional_Graph_NN_7batchsize_3inputchannel_9.pth'))

def eval_model_loss():
    dataset = GraphDataset()

    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, collate_fn=_collate_fn)
    lossfunction = nn.MSELoss()

    epoch_counter = 0
    for epoch in range(1):
        model.eval()
        epoch_counter += 1

        batch_counter = 0
        running_loss = 0.0
        for g, labels in dataloader:

            batch_counter += 1

            logits = model(g)

            loss = lossfunction(logits, labels)



            # print statistics
            running_loss += loss.item()
            if batch_counter >= (7395 // batchsize):  # print every 25 mini-batches
                print('TRAINING: Epoch %d Batch %d loss: %.3f' %
                      (epoch_counter, batch_counter, running_loss / batch_counter))





def get_unique_dist_g(LEP, iter=None):
    # g1, g2 = stats.uniform(loc=-pi/2, scale=pi).rvs(size=1)[0]
    expr = Expression(("1.0" + " * (cos( theta ) * meanX - sin( theta ) *meanY)",
                       "1.0" + " * (sin( theta ) * meanX + cos( theta ) * meanY)"),
                      meanX=0, meanY=-5000, theta=0.0, degree=1)

    # expr.theta = stats.uniform(loc=-pi/2, scale=pi).rvs(size=1, random_state=iter)[0]
    expr.theta = stats.truncnorm((-pi / 2) / 0.3, (pi / 2) / 0.3, loc=0, scale=0.3).rvs(size=1)[0]
    # a = expr(0)[0]
    # b = expr(0)[1]
    # #a, b = stats.uniform(loc=-1, scale=2).rvs(size=1)[0]
    # plt.quiver(*origin, a, b)
    # plt.show()

    LEP.g = expr
    return LEP


def sample_opti_sequenz_graph(iter=None, maxIteration=501):
    # initialize LEP
    LEP = LEProblem(bridge2Dstochastic())
    tau_adapter = tauadaptor(LEP.get_tau())
    mesh_adapter = meshadaptor(mesh_0=LEP.mesh, CVaR=LEP.CVaR)

    IterationStep = 0
    get_new_graph_iter = 0

    iters = []
    taus = []
    taus.append(LEP.tau_0)
    gammas = []
    es = []
    controls = []


    for k in range(maxIteration):
        IterationStep += 1
        get_new_graph_iter += 1
        iters.append(IterationStep)
        print("\nIter: {IterationStep}".format(IterationStep=IterationStep))

        # solving
        SE = LEP.SE(LEP.u, LEP.phi_n[0], LEP.v_u, LEP.g)
        solve(lhs(SE) == rhs(SE), LEP.u_n, bcs=LEP.bcSE, solver_parameters={"linear_solver": "umfpack", "preconditioner": "default"}, form_compiler_parameters=None)

        # get u_k and phi_{k-1}
        if get_new_graph_iter >= 50:


            # get edges from fenics
            edge_from = []
            edge_to = []
            dofmap = LEP.F.dofmap()
            for cell in cells(LEP.mesh):
                finite_element_node_index = dofmap.cell_dofs(cell.index())
                edge_from += list(list(zip(*itertools.permutations(finite_element_node_index[:-1], 2)))[0])
                edge_to += list(list(zip(*itertools.permutations(finite_element_node_index[:-1], 2)))[1])

            # initialize graph
            g = dgl.graph((edge_from, edge_to))  # initialize Graph (nodes and edges)
            print("Graph details:", g)
            print("Len of node IDs:", g.nodes())

            # initialize graph
            g.ndata['value'] = torch.tensor([[0.5] for i in range(g.number_of_nodes())], dtype=torch.float32)
            g.ndata['u'] = torch.tensor([[0.001, 0.001] for i in range(g.number_of_nodes())], dtype=torch.float32)

            # get graph data from fenics
            phi_val = LEP.phi_n.vector()[:-1]
            ux, uy = LEP.u_n.split(deepcopy=True)
            ux_val = ux.vector()[:]
            uy_val = uy.vector()[:]




            for i in range(len(phi_val)):
                g.ndata['value'][i] = phi_val[i]
                g.ndata['u'][i] = torch.tensor([ux_val[i], uy_val[i]], dtype=torch.float32)


            if IterationStep > 100:
                fig = plt.figure()
                plot(LEP.phi_n[0])
                fig.suptitle('TruePhi_' + str(IterationStep))
                #plt.savefig('eval_plots/'+str(IterationStep) + 'TruePhi_(' + str(LEP.g(0)[0]) + ',' + str(LEP.g(0)[1]) + '.png')
                plt.show()

                fig = plt.figure()
                plot(predicton)
                fig.suptitle('PredictedPhi_'+ str(IterationStep))
                #plt.savefig('eval_plots/'+str(IterationStep) + 'PredictedPhi_(' + str(LEP.g(0)[0]) + ',' + str(LEP.g(0)[1]) + '.png')
                plt.show()

            model_out = model(g)[:,0].detach().numpy()

            predicton, t = LEP.phi_n.split(deepcopy=True)

            predicton.vector()[:] = model_out

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
    sample_opti_sequenz_graph( maxIteration=405)