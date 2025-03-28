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
import targets
from adds import make_folder, create_gif
from plotting import plot_particles, plot_all_paths, plotKL
from tqdm import tqdm
import pingouin as pg
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform

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
        speed_restart=True,
        gradient_restart=True,
        beta=0,  # constant damping parameter
        verbose=True,
        arrows=True,
        eps=.1,  # Wasserstein-2 regularization parameter
        N=500,  # number of particles
        max_time=10,  # max time horizon
        d=2,  # dimension of the particles
        subdiv=1000,  # number of subdivisions of [0, max_time]
        sigma=.1,  # Gaussian kernel parameter
        target=targets.skewed_Gaussian,  # target from targets.py
        kernel_choice='generalized_bilinear',
        A=np.eye(2)  # parameter for bilinear kernel
        ):
    tau = max_time / subdiv  # constant step size
    Y = np.zeros((N, d))  # initial acceleration
    V = np.zeros((N, d))  # initial velocities

    # intial distribution
    init_mean = np.array([0, 0])  # initial mean
    init_cov = np.eye(d)  # initial covariance

    # X, _ = make_moons(N, noise=.1, random_state=42)  # initial particles
    # X = sampling(N, bananas_density)
    # prior_name = 'bananas'
    # uniform time step size
    X = np.random.multivariate_normal(init_mean, init_cov, size=N)
    print(np.mean(X**2, axis=0))
    prior_name = 'Gaussian'
    # initialize quantities tracked along the flow
    means, covs, KLs = 3*[np.zeros(subdiv+1)]
    if d <= 2:
        if kernel_choice == 'Gaussian':
            folder_name = (f'N={N},d={d},mu_0={init_mean.flatten()},'
                           + f'Sigma_0={init_cov.flatten()},eps={eps},'
                           + f'max_time={max_time},subdiv={subdiv},tau={tau},'
                           + f'{target.name}_target,'
                           + f'speed_restart={speed_restart},'
                           + f'beta={beta},'
                           + f'gradient_restart={gradient_restart},'
                           + f'sigma={sigma},prior={prior_name},'
                           + f'{kernel_choice}'
                           )
        elif kernel_choice == 'generalized_bilinear':
            folder_name = (f'N={N},d={d},mu_0={init_mean.flatten()},'
                           + f'Sigma_0={init_cov.flatten()},eps={eps},'
                           + f'max_time={max_time},subdiv={subdiv},tau={tau},'
                           + f'{target.name}_target,'
                           + f'speed_restart={speed_restart},'
                           + f'beta={beta},'
                           + f'gradient_restart={gradient_restart},'
                           + f'prior={prior_name},{kernel_choice}'
                           )
        else:
            print('No valid kernel choice selected!')
    else:
        folder_name = (f'N={N},d={d},eps={eps},'
                       + f'max_time={max_time},subdiv={subdiv},tau={tau},'
                       + f'{target.name}_target,speed_restart={speed_restart},'
                       + f'sigma={sigma},prior={prior_name}'
                       + f'gradient_restart={gradient_restart},'
                       + f'beta={beta},'
                       + f'sigma={sigma},prior={prior_name},A={A.flatten()}'
                       + f'{kernel_choice}'
                       )

    make_folder(folder_name)
    methods = ['ASVGD', 'SVGD', 'ULA', 'ULD', 'MALA']
    for m in methods:
        make_folder(folder_name+f'/{m}')

    # non-accelerated particles
    X_non = X.copy()
    Y_non = Y.copy()
    # Overdamped Langevin dynamics (ULA)
    X_over = X.copy()
    Y_over = Y.copy()
    X_MALA = X.copy()
    # Underdamped Langevin dynamics (ULD)
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
    Xs, Vs, Xs_non, Xs_over, Xs_under, Xs_MALA = [np.zeros((subdiv+1, N, d))
                                                  for _ in range(6)]

    # restart_counter: for computing acceleration parameter for each particle
    # will be reset to 1 when a restart is triggered.
    restart_counter = np.ones(N)  # each particle starts with counter=1
    alpha_k = np.zeros(N)         # acceleration parameter for each particle

    for k in tqdm(range(subdiv+1)):
        if target.name == 'Gaussian':
            if not pg.multivariate_normality(X, alpha=0.05).normal:
                print(f'Iter: {k}: Points are likely not Gaussian')
        if d == 1:
            empirical_cov = np.cov(X, rowvar=False) * np.eye(1)
        if d == 2:
            empirical_mean = np.mean(X, axis=0)
            empirical_cov = np.cov(X, rowvar=False)
            if target.mean is not None and target.cov is not None:
                means[k] = np.linalg.norm(empirical_mean - target.mean)
                covs[k] = np.linalg.norm(empirical_cov - target.cov)
        if N > 1 and d <= 2 and target.mean is not None and target.cov is not None:
            KLs[k] = KL(empirical_cov, target.cov, np.linalg.inv(target.cov),
                        empirical_mean, target.mean)
        # KDE for approximating KL loss via Monte Carlo (w/o normaliz. const)
        if N > d:
            try:
                X_KDE = gaussian_kde(X.T)
                KL_acc[k] = np.mean(np.log(X_KDE.evaluate(X.T)+1e-12) - np.log(
                                            np.array([target.density(x, y)
                                                      for x, y in X])+1e-12))
            except Exception:
                print('KDE for ASVGD failed')
            try:
                X_non_KDE = gaussian_kde(X_non.T)
                KL_non[k] = np.mean(np.log(X_non_KDE.evaluate(X_non.T)+1e-12)
                                    - np.log(np.array([target.density(x, y)
                                                       for x, y in X_non])+1e-12))
            except Exception:
                print('KDE for SVGD failed')
            try:
                X_over_KDE = gaussian_kde(X_over.T)
                KL_over[k] = np.mean(np.log(X_over_KDE.evaluate(X_over.T))
                                     - np.log(np.array([target.density(x, y)
                                                        for x, y in X_over])))
            except Exception:
                print('Some values for ULA are infs or NaNs')
            try:
                X_under_KDE = gaussian_kde(X_under.T)
                KL_under[k] = np.mean(np.log(X_under_KDE.evaluate(X_under.T)+1e-12)
                                      - np.log(np.array([target.density(x, y)
                                                        for x, y in X_under])+1e-12))
            except Exception:
                print('Some values for ULD are infs or NaNs')
            try:
                X_MALA_KDE = gaussian_kde(X_MALA.T)
                KL_MALA[k] = np.mean(np.log(X_MALA_KDE.evaluate(X_MALA.T)+1e-12)
                                     - np.log(np.array([target.density(x, y)                                            for x, y in X_MALA])+1e-12))
            except Exception:
                print('Some values for MALA are infs or NaNs')

        if d == 2 and plot and not k % 100:
            plot_particles(X, Y, k, folder_name, target, 'k',
                           'ASVGD', arrows)
            plot_particles(X_non, Y_non, f'non_acc_{k}', folder_name,
                           target, 'k', 'SVGD', arrows)
            plot_particles(X_over, Y_over, f'Overdamped_{k}', folder_name,
                           target, 'k', 'ULA', arrows)
            plot_particles(X_under, Y_under, f'Underdamped_{k}', folder_name,
                           target, 'k', 'ULD', arrows)
            plot_particles(X_MALA, None, f'MALA_{k}', folder_name,
                           target, 'k', 'MALA', arrows)
        Xs[k, :, :] = X
        Xs_over[k, :, :] = X_over
        Xs_under[k, :, :] = X_under
        Xs_MALA[k, :, :] = X_MALA
        Xs_non[k, :, :] = X_non
        # overdamped Langevin update (ULA) using Euler-Mayurama discretization
        determ = tau / gamma * np.apply_along_axis(target.score, 1, X_over)
        rand = sigma_0 * np.random.randn(N, d)
        Y_over = determ + rand
        X_over += Y_over
        # underdamped Langevin update using Euler-Mayurama discretization
        X_under += tau * Y_under
        Y_under = (1 - gamma / m * tau)*Y_under
        Y_under += tau / m * np.apply_along_axis(target.score, 1, X_under)
        Y_under += sigma_0 * np.random.randn(N, d)
        # MALA
        grad_current = np.apply_along_axis(target.score, 1, X_MALA)  # (N, d)
        # Generate proposal using Euler-Maruyama step:
        proposal_mean = X_MALA + tau/gamma * grad_current
        proposal = proposal_mean + sigma_0 * np.random.randn(N, d)  # (N, d)
        # Compute the ratio of target densities:
        # π(x)  exp(-U(x)), so the ratio is exp(-U(proposal) + U(current))
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
        # # SVGD i.e. non-accelerated Stein metric gradient flow
        h = sigma  # np.sqrt(0.5 * np.median(pairwise_dists) / np.log(X.shape[0]+1))
        # # compute the Gaussian kernel matrix
        if kernel_choice == 'Gaussian':
            K_non = np.exp(- squareform(pdist(X_non))**2 / (2 * h**2))
            Y_non = 1/N * ((np.diag(K_non.sum(axis=1)) - K_non) @ X_non / sigma**2
                           + K_non @ np.apply_along_axis(target.score, 1, X_non))
        elif kernel_choice == 'generalized_bilinear':
            K_non = X_non @ A @ X_non.T + np.ones((N, N))
            Y_non = X_non @ A + 1 / N * K_non @ np.apply_along_axis(target.score, 1, X_non)
        X_non += tau * Y_non
        # accelerated SVGD update positions
        K = np.exp(- squareform(pdist(X))**2 / (2 * sigma**2))  # kernel matrix
        # X += np.sqrt(tau) * Y  # predictor for X
        V = N * np.linalg.solve(K + N * eps * np.eye(N), Y)
        Vs[k, :, :] = V
        # adaptive speed restart
        if speed_restart and k > 0:
            norm_diff_current = np.linalg.norm(X - Xs[k-1, :, :], axis=1)  # ||x_{k+1}^i - x_k^i||
            norm_diff_prev = np.linalg.norm(Xs[k-1, :, :] - Xs[k-2, :, :], axis=1)  # ||x_k^i - x_{k-1}^i||
            # For each particle, check if current step is smaller than previous
            mask = norm_diff_current < norm_diff_prev
            # If yes, restart for that particle; else, increment its counter
            restart_counter[mask] = 0
            restart_counter[~mask] += 1
            alpha_k = (restart_counter) / (restart_counter + 3)
        elif not speed_restart:
            alpha_k = beta*np.ones(N)
        if gradient_restart:
            B = np.apply_along_axis(target.score, 1, X) + X  # (N x d)
            # First term: sum_{i,j} K[i,j] * (f(X_i) + X_i) dot V_j 
            term1 = np.sum((B.dot(V.T)) * K)
            # Second term: sum_{i,j} K[i,j] * (X_j dot V_j).
            # Note: For fixed j, X_j dot V_j does not depend on i, so:
            term2 = np.sum(np.sum(K, axis=0) * np.sum(X * V, axis=1))
            phi = term1 - term2
            if phi < 0:
                restart_counter = np.ones(N)
                print('gradient restart triggered')
        # ASVGD
        # K = np.exp(- squareform(pdist(X))**2 / (2 * sigma**2))  # kernel matrix
        # Xtilde = X + np.sqrt(tau) * Y  # predictor for X
        # Ktilde = np.exp(- squareform(pdist(Xtilde))**2 / (2 * sigma**2))  # kernel matrix predictor
        Y = alpha_k[:, None] * Y  # predictor for Y
        W = K @ np.multiply(V @ V.T, K) - np.multiply(K@V@V.T, K) + N * K
        # Z = np.multiply(K, V @ V.T) - K @ np.diag(np.sum(V**2, axis=1))
        # W += 2 * eps * N * np.sqrt(tau) * Z
        W_laplacian = np.diag(W.sum(axis=1)) - W
        if kernel_choice == 'generalized_bilinear':
            K = X @ A @ X.T + np.ones((N, N))
            V = N * np.linalg.solve(K + N * eps * np.eye(N), Y)
            Y += np.sqrt(tau) / N * K @ np.apply_along_axis(target.score, 1, X)
            Y += np.sqrt(tau) * (1 + N**(-2) * np.trace(V.T @ K @ V)) * X @ A
        else:
            Y += np.sqrt(tau) / N * np.apply_along_axis(target.score, 1, X)
            Y += np.sqrt(tau) / (2 * N**2 * sigma**2) * W_laplacian @ X
            #  + eps * 3 / (np.sqrt(tau) * (k + 1)) * V
        # corrector step
        # corrector = np.linalg.pinv(np.eye(N) - eps * np.linalg.inv(Ktilde / N + eps * np.eye(N)))
        # Y = corrector @ (Ytilde - eps * V)
        X += np.sqrt(tau) * Y
        # V = N * np.linalg.solve(K + N * eps * np.eye(N), Y)
        # early stopping
        if np.linalg.norm(np.apply_along_axis(target.score, 1, X)) < N * 1e-5:
            print('Early stopping triggered by ASVGD')
            break
        if np.linalg.norm(np.apply_along_axis(target.score, 1, X_non)) < N * 1e-5:
            print('Early stopping triggered by SVGD')
            break
        if np.linalg.norm(np.apply_along_axis(target.score, 1, X_MALA)) < N * 1e-5:
            print('Early stopping triggered by MALA')
            break
        if np.linalg.norm(np.apply_along_axis(target.score, 1, X_under)) < N * 1e-5:
            print('Early stopping triggered by ULD')
            break
        if np.linalg.norm(np.apply_along_axis(target.score, 1, X_over)) < N * 1e-5:
            print('Early stopping triggered by ULA')
            break
    if plot:
        for m in methods:
            create_gif(folder_name+f'/{m}', f'{folder_name}/{folder_name}_{m}.gif')

    # plotting quantities along the flow
    if d <= 2 and plot:
        plt.plot(means, label='deviation from mean')
        plt.plot(covs, label='deviation from cov')
        plt.plot(KLs, label='KL between empirical cov matrix and target cov')
        plt.title(f'N ={N}, d = {2}, eps = {eps}, tau = {tau}')
        plt.legend()
        plt.yscale('log')
        plt.savefig(f'{folder_name}/{folder_name}_loss.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    if plot:
        # plot KL loss
        plotKL(k, KL_acc, KL_non, KL_over, KL_under, KL_MALA,
               target.lnZ, folder_name)
        # plot paths
        plot_all_paths(k, Xs, Xs_non, Xs_under, Xs_over, Xs_MALA,
                       target, folder_name)

    return X


acc_Stein_Particle_Flow()
