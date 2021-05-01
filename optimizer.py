from fenics import *
from Linear_Elasticity_solver_stochastic import *
from adaptivity.tau_adaptivity import tauadaptor
from adaptivity.mesh_adaptivity import meshadaptor

from problems.bridge2Dstochastic import *
from problems.bridge2D import *

import time as tm
import matplotlib.pyplot as plt
from math import ceil

import multiprocessing

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

    #print("Phi_n as Vector:", LEP.phi_next.vector()[:])
    #print("Länge von Phi:", len(LEP.phi_next.vector()[:]))

    J_eps = LEP.J_eps(LEP.u_n, LEP.phi_n[0])

    return J_eps


def CVaR_condition(LEP):
    """
    check if ∫_(Γ_g) g u ds - t > 0
    :param LEP: stochastic Linear Elasticity Problem equipped the conditional value at risk (CVaR)
    :return: True or False
    """
    return (assemble(dot(LEP.g, LEP.u_n) * LEP.ds(2)) - LEP.get_t()) > 0


def p_CVaR(LEP, condition):
    """
    Drive solution (p) of the adjoint equation of an stochastic Linear Elasticity Problem equipped the conditional value at risk (CVaR)
    as an risk measure. The CVaR depends on risk t for a given probability ß.
    :param LEP: stochastic Linear Elasticity Problem equipped the conditional value at risk (CVaR)
    :return: Solution of the adjoint equation of an stochastic ELP
    """
    # define p(ß)
    if not condition: #∫_(Γ_g) g u ds - t <= 0
        return 0
    else:
        return LEP.u_n.vector().get_local()*1/(1-LEP.beta) # p_n = (1- ß)^-1 u_n.....self.alpha / ( 1 - self.beta ) * u


def CVaR_solve(LEP):
    """
    :param LEP: stochastic Linear Elasticity Problem equipped the conditional value at risk (CVaR)
    :return: Solution corresponding state SE(w), adjoint AE(w) and gradient GE(w) equation of the given LEP for one arbitrary but fixed w \in Omega
    """
    # solve state equation
    SE = LEP.SE(LEP.u, LEP.phi_n[0], LEP.v_u, LEP.g)
    solve(lhs(SE) == rhs(SE), LEP.u_n, bcs=LEP.bcSE, solver_parameters={"linear_solver": "umfpack", "preconditioner": "default"}, form_compiler_parameters=None)

    condition = CVaR_condition(LEP)

    # solve adjoint equation
    LEP.p_n.vector()[:] = p_CVaR(LEP, condition)

    # solve gradient equation [paper (2.15)]
    GE = LEP.GE_CVaR(LEP.phi, LEP.phi_n, LEP.v_phi, LEP.p_n, LEP.u_n, LEP.tau, LEP.gamma)
    solve(lhs(GE) == rhs(GE), LEP.phi_next, bcs=None, solver_parameters={"linear_solver": "umfpack", "preconditioner": "default"}, form_compiler_parameters=None)

    LEP.project_phi()

    J_eps = LEP.J_eps(LEP.u_n, LEP.phi_n[0])

    return J_eps, condition

def compliance(LEP):
    return assemble(dot(LEP.g, LEP.u_n) * LEP.ds(2))


def random_solve(random_state=None):
    """
    Works only in combination with monte_carlo_solve() which defines a global LEM (mc_problem) equipped the conditional value at risk (CVaR)
    as an risk measure. Simulate and solve new u(w), p(w), phi(w) with every call.
    :param random_state: Fix random state witch can be set for comparability
    :return: Sampel u(w), p(w), phi(w) of the given stochastic LEP and solve (CVaR_solve()) corresponding state, adjoint and gradient equation
    """
    # sparse copy of global variable (monte carlo problem)
    side_problem = ObjectWrapper(mc_problem)
    side_problem.sample(random_state)

    side_problem.u_n = Function(mc_problem.V)
    side_problem.u_n.assign(mc_problem.u_n)

    side_problem.p_n = Function(mc_problem.V)
    side_problem.p_n.assign(mc_problem.p_n)

    side_problem.phi_next = Function(mc_problem.F)
    side_problem.phi_next.assign(mc_problem.phi_next)

    if side_problem.CVaR:
        J_eps, condition = CVaR_solve(side_problem)
    else:
        J_eps = deterministic_solve(side_problem)

    return side_problem.u_n.vector().get_local(), side_problem.p_n.vector().get_local(), side_problem.phi_next.vector().get_local(), J_eps, condition



def monte_carlo_solve(LEP):
    """
    Sample u(w) und solve (random_solve) corresponding state, adjoint and gradient equation
    :param LEP: tochastic Linear Elasticity Problem equipped the conditional value at risk (CVaR)
    :return: Update u,p,phi by the mean of solutions samples u(w),p(w),phi(w) for next gradient step
    """
    # save problem as global variable
    global mc_problem
    mc_problem = LEP

    seeds = np.arange(LEP.nSamples) + 1 #+ LEP.quasiMClastSeed
    LEP.quasiMClastSeed = seeds.max()


    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() -1)
    samples = pool.map(random_solve, seeds)


    # collect samples
    samples = np.array(samples)
    samplesSorted = samples[samples[:, 3].argsort(axis=0)]

    u_samps = np.array(samplesSorted[:, 0])
    p_samps = np.array(samplesSorted[:, 1])
    phi_samps = np.array(samplesSorted[:, 2])
    J_eps_samps = np.array(samplesSorted[:, 3])
    # condition_samps = np.array(samplesSorted[:,4]) # needed for CVAR




    # calculate mean based on samples and Beta
    tailN = int(ceil(len(J_eps_samps) * (1 - LEP.beta)))
    LEP.u_n.vector()[:] = np.mean(u_samps[-tailN:], axis=0)
    LEP.p_n.vector()[:] = np.mean(p_samps[-tailN:], axis=0)
    LEP.phi_next.vector()[:] = np.mean(phi_samps[-tailN:], axis=0)

    LEP.project_phi()


    #print("Phi_n as Vector:", LEP.phi_next.vector()[:])
    #print("Länge von Phi:", len(LEP.phi_next.vector()[:]))

def LE_optimzation(problem, tau_adapter, maxIteration=500, plot_parameters=True, plot_steps=True, stoch=True, plot_every=10):
    ConvergenceIndicator = False
    iters = []
    times = []
    taus_t = []
    taus = []
    taus.append(problem.tau_0)
    gammas = []
    es = []
    es_t = []
    controls = []
    Js = []
    compliance_list = []
    t_list = []


    IterationStep = 0
    iter_step_for_plot = 98
    mesh_count = 1
    conv_min_iter = 50

    if problem.stochastic and stoch:
        problem.sample(0)


    while not ConvergenceIndicator:
        # solve state, adjoint and gradient equation
        iter_step_for_plot +=1
        IterationStep += 1
        iters.append(IterationStep)
        print("\nIter: {IterationStep}".format(IterationStep=IterationStep))
        StartTime = tm.time()
        print("solving...")
        if problem.stochastic and stoch:
            monte_carlo_solve(problem)
        else:
            deterministic_solve(problem)

        # collect tau, gamma, realtive change and t
        if problem.stochastic and stoch and problem.CVaR:
            tau_n = problem.get_tau_stoch()
            taus.append(tau_n[0])
            taus_t.append(tau_n[1])

            e_n = problem.get_e_n_stoch()[0]
            e_n_t = problem.get_e_n_stoch()[1]
            es.append(e_n)
            es_t.append(e_n_t)
            t_list.append(problem.get_t())
        else:
            tau_n = problem.get_tau()
            taus.append(tau_n)

            e_n = problem.get_e_n()
            es.append(e_n)

        #collect convergence control value
        control = problem.get_control_change()
        if problem.stochastic and stoch:
            controls.append(control / tau_n[0])
        else:
            controls.append(control / tau_n)

        gammas.append(problem.gamma(0))

        # drive J
        J = assemble(problem.J(problem.u_n, problem.phi_next[0], problem.gamma))
        Js.append(J)
        print("J=", Js[-1])
        compliance_list.append(compliance(problem))

        # adapt tau
        if problem.stochastic and problem.CVaR:
            problem.update_tau_stoch(tau_adapter.nextTau(problem.get_e_n_stoch()))
        else:

            problem.tau.vector()[:] = tau_adapter.nextTau(e_n)[0]


        # adapt gamma
        problem.new_adaptGamma()

        if iter_step_for_plot >= plot_every:
            plot(problem.phi_next[0] )
            plt.show()
            iter_step_for_plot = 0

        # update mesh if converged
        if problem.stochastic:
            NewMeshIndicator, new_mesh = mesh_adapter.update(problem.phi_next, problem.u_n, controls[-1], problem.mesh)
        else:
            NewMeshIndicator, new_mesh = mesh_adapter.update(problem.phi_next, problem.u_n, controls[-1], problem.mesh)

        conv_min_iter += 1
        if NewMeshIndicator:
            mesh_count += 1
            conv_min_iter = 0
            LEP.updateMesh(new_mesh)
            control = 1000

            if plot_steps:
                plt.figure()
                plot(problem.mesh)
                plt.show()
                plt.figure()
                plot(problem.phi_next[0])
                plt.show()
                plt.figure()
                plot(problem.u_n)
                plt.show()


        # update phi
        problem.phi_last.assign(problem.phi_n)
        problem.phi_n.assign(problem.phi_next)

        EndTime = tm.time()
        times.append(EndTime - StartTime)
        if IterationStep >= maxIteration:

            print("reach iteration limit")
            ConvergenceIndicator = True

            if plot_parameters:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                ax1.plot(iters, taus[1:])
                ax1.set_title("τ per iteration")
                ax3.plot(iters, gammas)
                ax3.set_title("γ per iteration")
                ax2.plot(iters, times)
                ax2.set_title("computing time per iteration")
                ax4.plot(iters, Js)
                ax4.set_title("J^eps(φ) per iteration")
                plt.show()
                #plt.savefig('opti_plots/matriken.png')


        if control < 1 and conv_min_iter >= 50 and mesh_count >= 4:  # Kovergenz
            ConvergenceIndicator = True
            tac()
            print("converge")
            print("J=", Js[-1])
            print("complince:", compliance(LEP))

            if plot_parameters:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                ax1.plot(iters, taus[1:])
                ax1.set_title("τ per iteration")
                ax3.plot(iters, gammas)
                ax3.set_title("γ per iteration")
                ax2.plot(iters, times)
                ax2.set_title("computing time per iteration")
                ax4.plot(iters, Js)
                ax4.set_title("J^eps(φ) per iteration")
                plt.show()
                #plt.savefig('opti_plots/matriken.png')


    return taus, es, es_t, times, controls, gammas, Js, compliance_list, t_list, problem


if __name__ == '__main__':
    tic()
    taus, es, es_t, times, controls, gammas, Js, compliance_list, t_list, problem = LE_optimzation(problem=LEP, tau_adapter=tau_adapter, maxIteration=750, plot_parameters=True, plot_steps=False, stoch=False, plot_every=100)
    tac()

    print("J=", Js[-1])
    print("complince:", compliance_list[-1])

    plt.figure()
    plot(problem.phi_next[0])
    plt.show()