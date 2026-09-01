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
from adds import make_folder, create_gif, KL
from plotting import plot_particles, plot_all_paths, plotKL
from tqdm import tqdm
import pingouin as pg
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform

# ---------------------------- kernel helpers ----------------------------


def gaussian_kernel(X, sigma):
    """
    Return K (N,N), A1 (N,N,d), A2 (N,N,d)
    A1[j,i,:] = grad_x k(X_j, X_i) = -(X_j - X_i)/sigma^2 * k
    A2[j,i,:] = grad_y k(X_j, X_i) =  (X_j - X_i)/sigma^2 * k
    """
    N, d = X.shape
    diffs = X[:, None, :] - X[None, :, :]        # (N,N,d)
    r2 = np.sum(diffs * diffs, axis=2)          # (N,N)
    K = np.exp(- r2 / (2.0 * sigma**2))         # (N,N)
    K_exp = K[..., None]                        # (N,N,1)
    A1 = - (diffs / (sigma**2)) * K_exp
    return K, A1, -A1


def bilinear_affine_kernel(X, A_param, add_const=True):
    """
    Parametrized bilinear-affine kernel k(x,y) = x^T A y + 1
    Returns K (N,N), A1 (N,N,d), A2 (N,N,d) with
      A1[j,i,:] = A @ X[i]   (grad wrt first arg)
      A2[j,i,:] = A @ X[j]   (grad wrt second arg)
    """
    N, d = X.shape
    A_mat = 0.5 * (A_param + A_param.T)
    XA = X @ A_mat.T                  # (N,d) rows = (A X[i])^T
    K = XA @ X.T                      # (N,N)
    if add_const:
        K = K + np.ones((N, N))
    A1 = np.broadcast_to(XA[None, :, :], (N, N, d)).copy()   # (N,N,d)
    A2 = np.broadcast_to(XA[:, None, :], (N, N, d)).copy()
    return K, A1, A2


def imq_kernel(X, c=1.0, beta=0.5):
    """
    IMQ kernel k(x,y) = (c^2 + ||x-y||^2)^{-beta}
    Returns K (N,N), A1 (N,N,d), A2 (N,N,d)
    Gradients:
      grad_x k = -2*beta * (x-y) / (c^2 + r2) * k
      grad_y k = +2*beta * (x-y) / (c^2 + r2) * k
    """
    N, d = X.shape
    diffs = X[:, None, :] - X[None, :, :]       # (N,N,d)
    r2 = np.sum(diffs * diffs, axis=2)          # (N,N)
    base = (c**2 + r2)                          # (N,N)
    K = base ** (-beta)                         # (N,N)
    base_exp = base[..., None]                  # (N,N,1)
    K_exp = K[..., None]                        # (N,N,1)
    A1 = - (2.0 * beta) * (diffs / base_exp) * K_exp
    return K, A1, -A1


def inverse_log_kernel(X, c=1.0, eps_stable=1e-12):
    """
    Inverse-log kernel (stable variant):
      k(x,y) = 1 / log(1 + c^2 + ||x-y||^2)
    Returns K (N,N), A1 (N,N,d), A2 (N,N,d)
    Gradients:
      grad_x k = - 2 * (x-y) / ( base * log(base)^2 )
      grad_y k = + 2 * (x-y) / ( base * log(base)^2 )
    where base = 1 + c^2 + r2
    """
    N, d = X.shape
    diffs = X[:, None, :] - X[None, :, :]        # (N,N,d)
    r2 = np.sum(diffs * diffs, axis=2)          # (N,N)
    base = 1.0 + c**2 + r2                      # (N,N)
    log_base = np.log(base + eps_stable)        # (N,N)
    K = 1.0 / (log_base + eps_stable)           # (N,N)
    denom = (base * (log_base**2 + eps_stable))[..., None]  # (N,N,1)
    A1 = - (2.0 * diffs) / denom                # (N,N,d)
    return K, A1, -A1


def smoothed_energy_kernel(X, r=1.0, eps_norm=1e-8):
    """
    Smoothed-energy kernel:
      k(x,y) = ||x||^r + ||y||^r - ||x-y||^r,
    where ||v|| := sqrt(v^T v + eps_norm) (smoothed norm).
    Returns K (N,N), A1 (N,N,d), A2 (N,N,d)

    Gradients:
      let s_x = ||x||_eps,  s_ij = ||x_j - x_i||_eps
      grad_x k(X_j, X_i) = r * X_j * s_xj^{r-2} - r * (X_j - X_i) * s_ji^{r-2}
      grad_y k(X_j, X_i) = r * X_i * s_xi^{r-2} + r * (X_j - X_i) * s_ji^{r-2}
    """
    N, d = X.shape
    # pairwise diffs, diffs[j,i,:] = X_j - X_i
    diffs = X[:, None, :] - X[None, :, :]           # (N,N,d)
    r2 = np.sum(diffs * diffs, axis=2)             # (N,N)

    # smoothed norms
    norms = np.sqrt(np.sum(X * X, axis=1) + eps_norm)    # (N,)
    norms_pow = norms ** (r - 2.0)                       # (N,)

    norms_pair = np.sqrt(r2 + eps_norm)                  # (N,N)
    norms_pair_pow = norms_pair ** (r - 2.0)             # (N,N)

    # kernel matrix K = ||x||^r + ||y||^r - ||x-y||^r
    # compute columns: px_j = ||X_j||^r
    px = norms ** r                                       # (N,)
    K = px[None, :] + px[:, None] - (norms_pair ** r)     # (N,N)

    # Gradients
    term = (r * X) * norms_pow[:, None]                # (N,d)

    norms_pair_pow_exp = norms_pair_pow[..., None]       # (N,N,1)
    A1 = term[:, None, :] - r * diffs * norms_pair_pow_exp
    A2 = term[None, :, :] + r * diffs * norms_pair_pow_exp

    return K, A1, A2


# Set random seed for reproducibility
np.random.seed(42)


def acc_Stein_Particle_Flow(
        plot=True,  # decide whether to plot the particles along the flow
        speed_restart=False,
        gradient_restart=False,
        beta=None,  # constant damping parameter
        verbose=True,
        arrows=True,
        eps=.1,  # Wasserstein-2 regularization parameter
        N=500,  # number of particles
        max_time=10,  # max time horizon
        d=2,  # dimension of the particles
        subdiv=1000,  # number of subdivisions of [0, max_time]
        sigma=.1,  # Gaussian kernel parameter
        target=targets.Gaussian,  # target from targets.py
        kernel_choice='inverse_log_kernel',
        A=np.eye(2)  # parameter for bilinear kernel
        ):
    tau = max_time / subdiv  # constant step size
    Y = np.zeros((N, d))  # initial acceleration
    V = np.zeros((N, d))  # initial velocities

    # intial distribution
    init_mean = np.ones(d)  # initial mean
    init_cov = np.ones(d) @ np.ones(d) + np.eye(d)  # initial covariance

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
                           + f'beta={beta},A={A.flatten()}'
                           + f'gradient_restart={gradient_restart},'
                           + f'prior={prior_name},{kernel_choice}'
                           )
        else:
            folder_name = (f'N={N},d={d},mu_0={init_mean.flatten()},'
                           + f'Sigma_0={init_cov.flatten()},eps={eps},'
                           + f'max_time={max_time},subdiv={subdiv},tau={tau},'
                           + f'{target.name}_target,'
                           + f'speed_restart={speed_restart},'
                           + f'beta={beta},'
                           + f'gradient_restart={gradient_restart},'
                           + f'prior={prior_name},{kernel_choice}'
                           )
        # else:
        #     print('No valid kernel choice selected!')
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
        if target.name == 'Gaussian' and kernel_choice == 'generalized_bilinear':
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
                                     - np.log(np.array([target.density(x, y)
                                                        for x, y in X_MALA])+1e-12))
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
        elif beta is not None:
            alpha_k = beta*np.ones(N)  # constant damping
        else:
            alpha_k = k / (k + 3) * np.ones(N)  # no restart
        if gradient_restart:
            if kernel_choice == 'generalized_bilinear':
                nablafX = -np.apply_along_axis(target.score, 1, X)
                phi = np.sum(V) * np.sum(nablafX)
                phi += - N * np.trace(V @ A @ X.T)
                phi += np.trace(V @ nablafX.T @ X @ A @ X.T)
                phi = - phi
            elif kernel_choice == 'Gaussian':
                B = np.apply_along_axis(target.score, 1, X) + X  # (N x d)
                # First term: sum_{i,j} K[i,j] * (f(X_i) + X_i) dot V_j
                term1 = np.sum((B.dot(V.T)) * K)
                # Second term: sum_{i,j} K[i,j] * (X_j dot V_j).
                term2 = np.sum(np.sum(K, axis=0) * np.sum(X * V, axis=1))
                phi = term1 - term2
            if phi < 0:
                restart_counter = np.ones(N)
                print('gradient restart triggered')
        plt.plot(alpha_k)
        plt.close()
        # ----------------------- ASVGD -----------------------
        if kernel_choice == 'Gaussian':
            K, A1, A2 = gaussian_kernel(X, sigma)
        elif kernel_choice == 'generalized_bilinear':
            K, A1, A2 = bilinear_affine_kernel(X, A)
        else:
            # fallback: 
            K, A1, A2 = inverse_log_kernel(X, sigma)

        V = N * np.linalg.solve(K + N * eps * np.eye(N), Y)   # (N,d)
        Vs[k, :, :] = V

        # Precompute commonly used matrices/vectors
        S = V @ V.T                     # (N,N)  S_{i,l} = <V_i, V_l>
        e = np.ones(N)                  # (N,)
        w = (S * K) @ e                 # (N,)
        M = S @ K.T                     # (N,N)

        # A2. shape = (N, N, d) with axes (j,i,m)
        b = np.einsum('jim->jm', A2)             # (N,d)   b_j = sum_i A2[j,i,:]
        term1 = np.einsum('jim,i->jm', A2, w)    # (N,d)   term1_j = sum_i A2[j,i,:] * w_i
        term2 = np.einsum('jim,ij->jm', A1, M)   # (N,d)   term2_j = sum_i M_{i,j} * A1[j,i,:]
        r = np.einsum('il,lim->im', S, A2)       # (N,d)   r_i = sum_l S_{i,l} * A2[l,i,:]
        term3 = K @ r                            # (N,d)
        score = np.apply_along_axis(target.score, 1, X)   # (N,d)

        Y = alpha_k[:, None] * Y \
            + (np.sqrt(tau) / N) * b \
            + (np.sqrt(tau) / N) * (K @ score) \
            + (np.sqrt(tau) / (N**2)) * (term1 + term2 - term3)

        X = X + np.sqrt(tau) * Y

        if np.linalg.norm(np.apply_along_axis(target.score, 1, X)) < N * 1e-5:
            print('Early stopping triggered by ASVGD')
            break
        # ---------------- end ASVGD block ------------------

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

    # plot KL loss
    plotKL(k, KL_acc, KL_non, KL_over, KL_under, KL_MALA,
           target.lnZ, folder_name)
    # plot paths
    plot_all_paths(k, Xs, Xs_non, Xs_under, Xs_over, Xs_MALA,
                   target, folder_name)

    np.save(f'{folder_name}/ASVGD_particles', Xs)
    np.save(f'{folder_name}/SVGD_particles', Xs_non)
    np.save(f'{folder_name}/ULD_particles', Xs_under)
    np.save(f'{folder_name}/ULA_particles', Xs_over)
    np.save(f'{folder_name}/MALA_particles', Xs_MALA)
    np.save(f'{folder_name}/ASVGD_KL', KL_acc)
    np.save(f'{folder_name}/SVGD_KL', KL_non)
    np.save(f'{folder_name}/ULD_KL', KL_under)
    np.save(f'{folder_name}/ULA_KL', KL_over)
    np.save(f'{folder_name}/MALA_KL', KL_MALA)

    return Xs, Xs_non, Xs_under, Xs_over, Xs_MALA, V, KL_acc, KL_non, KL_over, KL_under, KL_MALA


_, _, _, _, _, _, KL_acc, KL_non, KL_over, KL_under, KL_MALA = acc_Stein_Particle_Flow()
_, _, _, _, _, _, KL_acc_2, KL_non_2, _, _, _ = acc_Stein_Particle_Flow(A=8*np.linalg.inv(np.array([[3, -2], [-2, 3]])))

plt.plot(KL_acc, label=r'ASVGD, $A = \text{id}_2$')
plt.plot(KL_non, label=r'SVGD, $A = \text{id}_2$')
plt.plot(KL_acc_2, label=r'ASVGD, $A = 8Q^{-1}$')
plt.plot(KL_non_2, label=r'SVGD, $A = 8Q^{-1}$')
plt.plot(KL_over, label='ULA')
plt.plot(KL_under, label='ULD')
plt.plot(KL_MALA, label='MALA')
plt.legend()
plt.yscale('log')
plt.xlabel('Iterations')
plt.grid(which='both')
plt.minorticks_on()
plt.ylabel('MC approximation of KL')
plt.savefig('KL_comparison_8.png', dpi=300, bbox_inches='tight')
plt.show()