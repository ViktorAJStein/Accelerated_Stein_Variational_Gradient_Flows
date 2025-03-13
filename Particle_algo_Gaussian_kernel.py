# -*- coding: utf-8 -*-
"""
Particle algorithm to approximate Stein gradient flow with Gaussian kernel
K(x, y) = exp(-1/(2 sigma^2) * || x - y ||_2^2) w.r.t. the KL divergence

@author: Viktor Stein
"""
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.datasets import make_moons
# import scipy as sp
# from scipy.integrate import dblquad
from targets import skewed_Gaussian, GMM, bananas, U2, U3, U4
from adds import make_folder, create_gif
from plotting import plot_particles, plot_all_paths, plotKL
from tqdm import tqdm
import pingouin as pg
from scipy.stats import gaussian_kde

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
        N=100,  # number of particles
        max_time=100,  # max time horizon
        d=2,  # dimension of the particles
        subdiv=100,  # number of subdivisions of [0, max_time]
        sigma=.1,  # Gaussian kernel parameter
        target=skewed_Gaussian  # target object from custom class (from targets.py)
        ):
    tau = max_time / subdiv
    Y = np.zeros((N, d))  # initial velocities

    # intial distribution
    init_mean = np.zeros(d)  # initial mean
    init_cov = np.ones(d) @ np.ones(d) + np.eye(d)  # initial covariance

    # X, _ = make_moons(N, noise=.1, random_state=42)  # initial particles
    # X = sampling(N, bananas_density)
    # prior_name = 'bananas'
    # uniform time step size
    X = np.random.multivariate_normal(init_mean, init_cov, size=N)
    prior_name = 'Gaussian'
    # initialize quantities tracked along the flow
    means = np.zeros(subdiv+1)
    covs = np.zeros(subdiv+1)
    KLs = np.zeros(subdiv+1)
    if d <= 2:
        folder_name = (f'N={N},d={d},mu_0={init_mean.flatten()},'
                       + f'Sigma_0={init_cov.flatten()},eps={eps},'
                       + f'max_time={max_time},subdiv={subdiv},tau={tau},'
                       + f'{target.name}_target_restart={adaptive_restart},'
                       + f'sigma={sigma},prior={prior_name}'
                       )
    else:
        folder_name = (f'N={N},d={d},eps={eps},'
                       + f'max_time={max_time},subdiv={subdiv},tau={tau},'
                       + f'{target.name}_target_restart={adaptive_restart},'
                       + f'sigma={sigma},prior={prior_name}'
                       )

    make_folder(folder_name)
    methods = ['accelerated', 'non-accelerated', 'overdamped',
               'underdamped', 'MALA']
    for m in methods:
        make_folder(folder_name+f'/{m}')

    # non-accelerated particles
    X_non = X.copy()
    Y_non = Y.copy()
    # Overdamped Langevin dynamics
    X_over = X.copy()
    Y_over = Y.copy()
    X_MALA = X.copy()
    # Underdamped Langevin dynamics
    X_under = X.copy()
    Y_under = Y.copy()
    # parameter for Langevin dynamics
    gamma = 1  # friction coefficient for Langevin dynamics
    temp = 1  # temperature * Boltzmann's constant
    m = 1  # mass of the particles
    sigma_0 = np.sqrt(2 * temp * tau / gamma)  # variance for the noise
    

    def q_density(y, x, sigma2):
        """
        Proposal density q(y|x) for MALA, which is Gaussian with mean
        x - (dt/gamma) * grad_U(x) and variance sigma2.
        y and x are arrays of shape (N, d).
        sigma2 is the variance sigma^2, not the std sigma.
        Returns: density evaluated for each particle (shape (N,))
        """
        d = x.shape[1]
        # Mean of the proposal
        mean = x + (tau/gamma) * np.apply_along_axis(target.score, 1, x)
        diff = y - mean
        norm_sq = np.sum(diff**2, axis=-1)
        coeff = 1.0 / ((2 * np.pi * sigma2)**(d/2))
        return coeff * np.exp(- norm_sq / (2 * sigma2))

    # history
    KL_acc, KL_non, KL_over, KL_under, KL_MALA = [np.zeros(subdiv+1)
                                                  for _ in range(5)]
    Xs, Xs_non, Xs_over, Xs_under, Xs_MALA = [np.zeros((subdiv+1, N, d))
                                              for _ in range(5)]

    X_prev = X.copy()
    L = []  # list of \| W - id \|_2
    # restart_counter is used to compute the acceleration parameter for each particle.
    # It will be reset to 1 when a restart is triggered.
    restart_counter = np.ones(N)  # each particle starts with counter=1
    alpha_k = np.zeros(N)         # acceleration parameter for each particle

    for k in tqdm(range(subdiv+1)):
        # KDE for approximating KL loss via Monte Carlo
        X_KDE = gaussian_kde(X.T)
        X_non_KDE = gaussian_kde(X_non.T)
        X_over_KDE = gaussian_kde(X_over.T)
        X_under_KDE = gaussian_kde(X_under.T)
        X_MALA_KDE = gaussian_kde(X_MALA.T)
        # approximate KL between particles and target (w/o normalization const)
        KL_acc[k] = np.mean(np.log(X_KDE.evaluate(X.T) / np.array([target.density(x, y) for x, y in X])))
        KL_non[k] = np.mean(np.log(X_non_KDE.evaluate(X_non.T) / np.array([target.density(x, y) for x, y in X_non])))
        KL_over[k] = np.mean(np.log(X_over_KDE.evaluate(X_over.T) / np.array([target.density(x, y) for x, y in X_over])))
        KL_under[k] = np.mean(np.log(X_under_KDE.evaluate(X_under.T) / np.array([target.density(x, y) for x, y in X_under])))
        KL_MALA[k] = np.mean(np.log(X_MALA_KDE.evaluate(X_MALA.T) / np.array([target.density(x, y) for x, y in X_MALA])))

        if d == 2 and plot and not k % 25:
            plot_particles(X, Y, k, folder_name, target.density, 'k',
                           'accelerated', arrows)
            plot_particles(X_non, Y_non, f'non_acc_{k}', folder_name,
                           target.density, 'k', 'non-accelerated', arrows)
            plot_particles(X_over, Y_over, f'Overdamped_{k}', folder_name,
                           target.density, 'k', 'overdamped', arrows)
            plot_particles(X_under, Y_under/100, f'Underdamped_{k}', folder_name,
                           target.density, 'k', 'underdamped', arrows)
            plot_particles(X_MALA, None, f'MALA_{k}', folder_name,
                           target.density, 'k', 'MALA', arrows)
        # save previous iters and concatenate history
        X_old = X_prev.copy()
        X_prev = X.copy()
        Xs[k, :, :] = X
        # overdamped Langevin update (ULA) using Euler-Mayurama discretization
        determ = tau / gamma * np.apply_along_axis(target.score, 1, X_over)
        rand = sigma_0 * np.random.randn(N, d)
        Y_over = determ + rand
        X_over += Y_over
        Xs_over[k, :, :] = X_over
        # underdamped Langevin update using Euler-Mayurama discretization
        X_under += tau * Y_under
        Y_under = (1 - gamma / m * tau)*Y_under
        Y_under += tau / m * np.apply_along_axis(target.score, 1, X_under)
        Y_under += sigma_0 * np.random.randn(N, d)
        Xs_under[k, :, :] = X_under
        # MALA
        grad_current = np.apply_along_axis(target.score, 1, X_MALA)  # (N, d)
        # Generate proposal using Euler-Maruyama step:
        proposal_mean = X_MALA + tau/gamma * grad_current
        proposal = proposal_mean + sigma_0 * np.random.randn(N, d)  # (N, d)
        # Compute the ratio of target densities:
        # π(x) ∝ exp(-U(x)), so the ratio is exp(-U(proposal) + U(current))
        up = np.array([target.density(row[0], row[1]) for row in proposal])
        down = np.array([target.density(row[0], row[1]) for row in X_MALA])
        target_ratio = up / down  # (N, )
        # Compute the proposal density ratio:
        # q(current|proposal) / q(proposal|current)
        q_forward = q_density(proposal, X_MALA, sigma_0**2)  # shape = (N, )
        q_backward = q_density(X_MALA, proposal, sigma_0**2)  # shape = (N, )
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
            if target.name == 'Gaussian':
                print(f'Iter: {k}: Points are likely not Gaussian')
        # kernel matrix
        sq_norms = np.sum(X**2, axis=1)
        D2 = sq_norms[:, None] + sq_norms[None, :] - 2 * np.dot(X, X.T)
        K = np.exp(-D2 / (2 * sigma**2))

        # non-accelerated
        sq_norms_non = np.sum(X_non**2, axis=1)
        D2_non = sq_norms_non[:, None] + sq_norms_non[None, :] - 2 * np.dot(X_non, X_non.T)
        K_non = np.exp(-D2_non / (2 * sigma**2))
        Y_non = 1 / N * ((np.diag(K_non.sum(axis=1)) - K_non) @ X_non / sigma**2 + K_non @ np.apply_along_axis(target.score, 1, X_non))
        X_non += tau * Y_non
        Xs_non[k, :, :] = X_non
        # accelerated
        try:
            V = np.linalg.pinv(K + N*eps*np.eye(N)) @ Y
        except Exception:
            print('Breaking due to noninvertibility of K + N*eps*eye(N)')
            break
        if d == 2:
            empirical_mean = np.mean(X, axis=0)
            empirical_cov = np.cov(X, rowvar=False)
            means[k] = np.linalg.norm(empirical_mean - target.mean)
            covs[k] = np.linalg.norm(empirical_cov - target.cov)
        if d == 1:
            empirical_cov = empirical_cov * np.eye(1)
        if d <= 2:
            KLs[k] = KL(empirical_cov, target.cov, np.linalg.inv(target.cov),
                        empirical_mean, target.mean)
        # two quantities for adaptive restart
        norm_diff_current = np.linalg.norm(X - X_prev, axis=1)  # ||x_{k+1}^i - x_k^i||
        norm_diff_prev = np.linalg.norm(X_prev - X_old, axis=1)  # ||x_k^i - x_{k-1}^i||

        # adaptive restart
        if adaptive_restart and k > 0:
            # if target.name == 'Gaussian' and KLs[k] - KLs[k - 1] > 0:
            #     if verbose:
            #         print(f'No KL-descent at iteration {k}, restarting momentum')
            #     restart_counter = 1
            # speed restart
            if target.name != 'Gaussian':
                # For each particle, check if the current step is smaller than the previous step
                mask = norm_diff_current < norm_diff_prev
                # If yes, restart for that particle; otherwise, increment its counter
                restart_counter[mask] = 1
                restart_counter[~mask] += 1
                # plt.plot(restart_counter)
                # plt.show()
            else:
                restart_counter += 1
            # acceleration parameter
            alpha_k = (restart_counter - 1) / (restart_counter + 2)
        else:
            alpha_k = (k - 1) / (k + 2)*np.ones(N)
        # update velocities
        Y = (1 - tau*alpha_k[:, None]) * Y
        W = K + N * (K @ np.multiply(K, V@V.T) - np.multiply(K@V@V.T, K))
        W_laplacian = np.diag(W.sum(axis=1)) - W
        Y += tau / (N * 2 * sigma**2) * (W_laplacian @ X + 2*sigma**2 * K @ np.apply_along_axis(target.score, 1, X))
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
        plt.savefig(f'{folder_name}/{folder_name}_loss.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    # plot KL loss
    plotKL(k, KL_acc, KL_non, KL_over, KL_under, KL_MALA, target.lnZ, folder_name)
    # plot paths
    plot_all_paths(k, Xs, Xs_non, Xs_under, Xs_over, Xs_MALA, folder_name)

acc_Stein_Particle_Flow()
