# -*- coding: utf-8 -*-
"""
Particle algorithm to approximate Stein gradient flow with Gaussian kernel
K(x, y) = exp(-1/(2 sigma^2) * || x - y ||_2^2) w.r.t. the KL divergence

@author: Viktor Stein
"""
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.datasets import make_moons
import scipy as sp
from targets import *
from adds import *
from plotting import *
from tqdm import tqdm
import pingouin as pg

# Set random seed for reproducibility
np.random.seed(42)


# KL divergence of N(mu, A) to N(nu, B)
def KL(A, B, B_inv, mu, nu):
    _, logdet_A = np.linalg.slogdet(A)
    _, logdet_B = np.linalg.slogdet(B)
    first = np.trace(B_inv @ A)
    second = logdet_B - logdet_A
    third = (nu - mu).T @ B_inv @ (nu - mu)
    d = A.shape[0]
    return 0.5 * (first - d + second + third)


def acc_Stein_Particle_Flow(
        plot=True,  # decide whether to plot the particles along the flow
        adaptive_restart=True,
        verbose=True,
        arrows=True,
        eps=0.01,  # regularization parameter
        N=250,  # number of particles
        max_time=100,  # max time horizon
        d=2,  # dimension of the particles
        subdiv=1000,  # number of subdivisions of [0, max_time]
        sigma=.1  # Gaussian kernel parameter
        ):
    tau = max_time / subdiv
    Y = np.zeros((N, d))  # initial velocities

    # intial distribution
    init_mean = np.zeros(d)  # initial mean
    init_cov = np.ones(d) @ np.ones(d) + np.eye(d)  # initial covariance

    # X, _ = make_moons(N, noise=.1, random_state=42)  # initial particles
    # X = sampling(N, bimodal2_density)
    # prior_name = 'bimodal_2'
    # uniform time step size
    X = np.random.multivariate_normal(init_mean, init_cov, size=N)
    prior_name = 'Gaussian'

    # target
    # nu = np.ones(d)  # target mean
    # Q = np.eye(d) * 5  # target covariance
    # Q_inv = np.linalg.inv(Q)  # inverse target covariance matrix

    # def target_score(x):
    #     return Q_inv @ (x - nu)

    # target_type = 'Gaussian'
    # target_mean = nu
    # target_cov = Q

    # def target_density(x, y):
    #     point = np.dstack((x, y))
    #     return sp.stats.multivariate_normal.pdf(point, mean=nu, cov=Q)

    # target_score = gauss_mix_grad
    # target_density = gauss_mix_density
    # target_type = 'non-Gaussian'
    # target_mean = np.zeros(2)
    # target_cov = np.eye(2) + 1/4*np.ones((2, 2))

    # target_score = U2_grad
    # target_density = U2_density
    # target_type = 'squiggly'
    # target_mean = np.zeros(2)
    # target_cov = np.eye(2) + 1/4*np.ones((2, 2))

    target_score = U3_grad
    target_density = U3_density
    target_type = 'squiggly2'
    target_mean = np.zeros(2)
    target_cov = np.eye(2) + 1/4*np.ones((2, 2))

    # target_score = U4_grad
    # target_density = U4_density
    # target_type = 'squiggly3'
    # target_mean = np.zeros(2)
    # target_cov = np.eye(2) + 1/4*np.ones((2, 2))

    # target_score = bimodal_grad
    # target_density = bimodal_density
    # target_type = 'non-Gaussian'
    # target_mean = np.zeros(2)
    # target_cov = np.array([[1.43, 0], [0, 9.125]])

    # initialize quantities tracked along the flow
    means = np.zeros(subdiv)
    covs = np.zeros(subdiv)
    KLs = np.zeros(subdiv)

    # print initial 
    if d <= 2:
        folder_name = (f'N={N},d={d},mu_0={init_mean.flatten()},'
                       + f'Sigma_0={init_cov.flatten()},eps={eps},'
                       + f'max_time={max_time},subdiv={subdiv},tau={tau},'
                       + f'{target_type}_target_restart={adaptive_restart},'
                       + f'sigma={sigma},prior={prior_name}'
                       )
    else:
        folder_name = (f'N={N},d={d},eps={eps},'
                       + f'max_time={max_time},subdiv={subdiv},tau={tau},'
                       + f'{target_type}_target_restart={adaptive_restart},'
                       + f'sigma={sigma},prior={prior_name}'
                       )

    make_folder(folder_name)
    methods = ['accelerated', 'non-accelerated', 'overdamped',
               'underdamped', 'MALA']
    for m in methods:
        make_folder(folder_name+f'/{m}')
    # restart_counter is used to compute the acceleration parameter.
    # It will be reset to 1 when a restart is triggered.
    restart_counter = 1

    # non-accelerated particles
    X_non = X.copy()
    # Overdamped Langevin dynamics
    X_over = X.copy()
    X_MALA = X.copy()
    # Underdamped Langevin dynamics
    X_under = X.copy()
    Y_under = Y.copy()
    # parameter for Langevin dynamics
    gamma = 1  # friction coefficient for Langevin dynamics
    temp = 1  # temperature * Boltzmann's constant
    m = 1  # mass of the particles
    sigma_0 = np.sqrt(2 * temp * tau / gamma)  # variance for the noise
    U = lambda x: -np.log(target_density(x[0], x[1]))

    def q_density(y, x, sigma2):
        """
        Proposal density q(y|x) for MALA, which is Gaussian with mean
        x - (dt/gamma) * grad_U(x) and variance sigma2.
        y and x are arrays of shape (N, d).
        Returns: density evaluated for each particle (shape (N,))
        """
        d = x.shape[1]
        # Mean of the proposal
        mean = x - (tau/gamma) * np.apply_along_axis(target_score, 1, x)
        diff = y - mean
        norm_sq = np.sum(diff**2, axis=-1)
        coeff = 1.0 / ((2 * np.pi * sigma2)**(d/2))
        return coeff * np.exp(- norm_sq / (2 * sigma2))

    # history
    Xs = np.zeros((subdiv, N, d))
    Xs_non = np.zeros((subdiv, N, d))
    Xs_over = np.zeros((subdiv, N, d))
    Xs_under = np.zeros((subdiv, N, d))
    Xs_MALA = np.zeros((subdiv, N, d))
    X_prev = X.copy()
    L = []
    for k in tqdm(range(subdiv)):
        if d == 2 and plot and not k % 25:
            plot_particles(X, Y, k, folder_name, target_density, 'k',
                           'accelerated', arrows)
            plot_particles(X_non, None, f'non_acc_{k}', folder_name,
                           target_density, 'b', 'non-accelerated', arrows)
            plot_particles(X_over, None, f'Overdamped_{k}', folder_name,
                           target_density, 'b', 'overdamped', arrows)
            plot_particles(X_under, Y_under, f'Underdamped_{k}', folder_name,
                           target_density, 'b', 'underdamped', arrows)
            plot_particles(X_MALA, None, f'MALA_{k}', folder_name,
                           target_density, 'b', 'MALA', arrows)
        # save previous iters and concatenate history
        X_old = X_prev.copy()
        X_prev = X.copy()
        Xs[k, :, :] = X
        # overdamped Langevin update (ULA) using Euler-Mayurama discretization
        X_over -= tau / gamma * np.apply_along_axis(target_score, 1, X_over)
        X_over += sigma_0 * np.random.rand(N, d)
        Xs_over[k, :, :] = X_over
        # underdamped Langevin update using Euler-Mayurama discretization
        X_under += tau * Y_under
        Y_under = (1 - gamma / m * tau)*Y_under
        Y_under -= tau / m * np.apply_along_axis(target_score, 1, X_under)
        Y_under += sigma_0 * np.random.rand(N, d)
        Xs_under[k, :, :] = X_under
        # MALA
        grad_current = np.apply_along_axis(target_score, 1, X_MALA)  # (N, d)
        # Generate proposal using Euler-Maruyama step:
        proposal_mean = X_MALA - tau/gamma * grad_current
        proposal = proposal_mean + sigma_0 * np.random.randn()  # shape = (N, d)
        # Compute the ratio of target densities:
        # π(x) ∝ exp(-U(x)), so the ratio is exp(-U(proposal) + U(current))
        up = np.array([target_density(row[0], row[1]) for row in proposal])
        down = np.array([target_density(row[0], row[1]) for row in X_MALA])
        target_ratio = up / down  # (N, )
        # Compute the proposal density ratio:
        # q(current|proposal) / q(proposal|current)
        q_forward = q_density(proposal, X_MALA, sigma_0)  # shape = (N, )
        q_backward = q_density(X_MALA, proposal, sigma_0)  # shape = (N, )
        proposal_ratio = q_backward / q_forward  # shape = (N, )
        # Acceptance probability for each particle
        alpha = np.minimum(1, target_ratio * proposal_ratio)
        # Accept or reject proposals
        u = np.random.rand(N)  # one uniform random number per particle
        accept = u < alpha  # boolean array, shape (N,)
        # Update: if accepted, move to proposal; otherwise, stay at current
        new_x = np.copy(X_MALA)
        new_x[accept] = proposal[accept]
        X_MALA = new_x
        Xs_MALA[k, :, :] = X_MALA
        # accelerated SVGD update positions
        X += tau * Y
        if not pg.multivariate_normality(X, alpha=0.05).normal:
            if target_type == 'Gaussian':
                print(f'Iter: {k}: Points are likely not Gaussian')
        # kernel matrix
        sq_norms = np.sum(X**2, axis=1)
        D2 = sq_norms[:, None] + sq_norms[None, :] - 2 * np.dot(X, X.T)
        K = np.exp(-D2 / (2 * sigma**2))

        # non-accelerated
        sq_norms_non = np.sum(X_non**2, axis=1)
        D2_non = sq_norms_non[:, None] + sq_norms_non[None, :] - 2 * np.dot(X_non, X_non.T)
        K_non = np.exp(-D2_non / (2 * sigma**2))
        phi = (np.diag(K_non.sum(axis=1)) - K_non) @ X_non - K_non @ np.apply_along_axis(target_score, 1, X_non) / sigma**2
        X_non += tau / N * phi
        Xs_non[k, :, :] = X_non

        try:
            V = np.linalg.pinv(K + N*eps*np.eye(N)) @ Y
        except Exception:
            break
        if d == 2:
            empirical_mean = np.mean(X, axis=0)
            empirical_cov = np.cov(X, rowvar=False)
            means[k] = np.linalg.norm(empirical_mean - target_mean)
            covs[k] = np.linalg.norm(empirical_cov - target_cov)
        if d == 1:
            empirical_cov = empirical_cov * np.eye(1)
        if d <= 2:
            KLs[k] = KL(empirical_cov, target_cov, np.linalg.inv(target_cov),
                        empirical_mean, target_mean)
        # two quantities for adaptive restart
        norm_diff_current = np.linalg.norm(X - X_prev)  # ||x_{k+1} - x_k||
        norm_diff_prev = np.linalg.norm(X_prev - X_old)  # ||x_k - x_{k-1}||

        # adaptive restart
        if adaptive_restart and k > 0:
            if target_type == 'Gaussian' and KLs[k] - KLs[k - 1] > 0:
                if verbose:
                    print(f'No KL-descent at iteration {k}, restarting momentum')
                restart_counter = 1
            # speed restart
            elif target_type != 'Gaussian' and norm_diff_current < norm_diff_prev:
                if verbose:
                    print(f'No norm descent of iterates at iteration {k}, restarting momentum')
                restart_counter = 1
            else:
                restart_counter += 1
            # acceleration parameter
            alpha_k = (restart_counter - 1) / (restart_counter + 2)
        else:
            alpha_k = (k - 1) / (k + 2)
        # update velocities
        Y = (1 - tau*alpha_k) * Y
        W = K + N * (K @ np.multiply(K, V@V.T) - np.multiply(K@V@V.T, K))
        W_laplacian = np.diag(W.sum(axis=1)) - W
        Y += tau / (N * 2 * sigma**2) * (W_laplacian @ X - 2*sigma**2 * K @ np.apply_along_axis(target_score, 1, X))
        L.append(np.linalg.norm(W - np.eye(N)))

        # early stopping for efficiency
        # if KLs[k] < 1e-7:
        #     print('Functional values is below 1e-7, stopping iteration.')
        #     break

    # plt.plot(L, label='\| W - Id \|')
    # plt.legend()
    # plt.show()

    if d == 2:
        print('Final empirical mean and covariance matrix:'
              + f' {empirical_mean}, {empirical_cov}')
        plot_particles(X, Y, k, folder_name, target_density, 'k',
                       'accelerated', arrows)
        plot_particles(X_non, None, f'non_acc_{k}', folder_name, target_density,
                       'b', 'non-accelerated', arrows)

    for m in methods:
        create_gif(folder_name+f'/{m}', f'{folder_name}/{folder_name}_{m}.gif')

    # plotting quantities along the flow
    if d <= 2:
        plt.plot(means, label='deviation from mean')
        plt.plot(covs, label='deviation from cov')
        plt.plot(KLs, label='KL between empirical cov matrix and target cov')
        plt.title(f'N ={N}, d = {2}, eps = {eps}, tau = {tau}')
        plt.legend()
        plt.yscale('log')
        plt.savefig(f'{folder_name}/{folder_name}_loss.svg',
                    dpi=300, bbox_inches='tight')
        plt.show()

    plot_paths(Xs, folder_name)
    plot_paths(Xs_non, folder_name, 'non-acc')
    plot_paths(Xs_under, folder_name, 'underdamped')
    plot_paths(Xs_over, folder_name, 'overdamped')
    plot_paths(Xs_MALA, folder_name, 'MALA')

acc_Stein_Particle_Flow()
