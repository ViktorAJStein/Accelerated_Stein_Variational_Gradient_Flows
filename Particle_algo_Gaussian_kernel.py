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
from scipy.stats import gaussian_kde
from tqdm import tqdm
import os
from PIL import Image
import pingouin as pg

# Set random seed for reproducibility
np.random.seed(42)


def sampling(n_samples, pdf):
    # Define the support (bounding box) for sampling
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    # An estimate of the maximum value of the pdf in the region
    max_pdf = 3
    samples = []
    while len(samples) < n_samples:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        u = np.random.uniform(0, max_pdf)
        if u < pdf(x, y):
            samples.append((x, y))
    return np.array(samples)


def get_timestamp(file_name):
    return int(file_name.split('_')[-1].split('.')[0])


def create_gif(image_folder, output_gif):
    images = []
    try:
        for filename in sorted(os.listdir(image_folder), key=get_timestamp):
            if filename.endswith(".png"):
                img = Image.open(os.path.join(image_folder, filename))
                images.append(img)

        if images:
            images[0].save(
                output_gif,
                save_all=True,
                append_images=images[1:],
                duration=np.floor(10000/len(images)),  # in milliseconds
                loop=0  # numbero f loops, 0 means infinite loop,
            )
    finally:
        for img in images:
            img.close()
    print('Gif created successfully!')


def make_folder(name):
    try:
        os.mkdir(name)
        print(f"Folder '{name}' created successfully.")
    except FileExistsError:
        print(f"Folder '{name}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}.")


def KL(A, B, B_inv, mu, nu):
    _, logdet_A = np.linalg.slogdet(A)
    _, logdet_B = np.linalg.slogdet(B)
    first = np.trace(B_inv @ A)
    second = logdet_B - logdet_A
    third = (nu - mu).T @ B_inv @ (nu - mu)
    d = A.shape[0]
    return 0.5 * (first - d + second + third)


def plot_particles(particles, velocities, label, folder_name, target, c, acc, arrows, KDE=False):
    m1, m2 = particles[:, 0], particles[:, 1]
    xmin = -8  # m1.min() - 1.0
    xmax = 8  # m1.max() + 1.0
    ymin = -8  # m2.min() - 1.0
    ymax = 8  # m2.max() + 1.0
    # Perform a kernel density estimate on the data:
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    fig, ax = plt.subplots()
    ax = plt.gca()
    if KDE:
        try:
            positions = np.vstack([X.ravel(), Y.ravel()])
            values = np.vstack([m1, m2])
            kernel = gaussian_kde(values)
            Z = np.reshape(kernel(positions).T, X.shape)
            ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
                      extent=[xmin, xmax, ymin, ymax])
        except Exception:
            pass
    if arrows and velocities is not None:
        plt.quiver(m1, m2, velocities[:, 0], velocities[:, 1],
                   angles='xy', scale_units='xy', scale=.01, alpha=.2)
    ax.plot(m1, m2, '.', markersize=2, color=c)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_aspect('equal', adjustable='box')
    # plot contour lines of target density
    T = target(X, Y)
    plt.contour(X, Y, T, levels=5, colors='black', alpha=.2)
    plt.title(f'Iteration {label}')
    plt.grid('True')
    plt.savefig(f'{folder_name}/{acc}/{folder_name}_{label}.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_paths(Xs, folder_name, add=''):
    N = Xs.shape[1]
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, N)]
    plt.figure(figsize=(10, 8))
    for i in range(N):
        # Plot the trajectory with transparency (alpha).
        plt.plot(Xs[:, i, 0], Xs[:, i, 1], c=colors[i], alpha=0.2)
        # Mark the starting point with a circle marker.
        plt.plot(Xs[0, i, 0], Xs[0, i, 1], marker='o', c=colors[i], ms=5)
        # Mark the endpoint with a square marker.
        plt.plot(Xs[-1, i, 0], Xs[-1, i, 1], marker='s', c=colors[i], ms=5)
    plt.title(f'Particle Trajectories {add} \n'
              + folder_name + '\n'
              + ' Starting points are circles, endpoints are squares')
    plt.grid(True)
    plt.xlim([-8, 8])
    plt.ylim([-5, 5])
    plt.savefig(f'{folder_name}/{folder_name}_{add}_paths.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def acc_Stein_Particle_Flow(
        plot=True,  # decide whether to plot the particles along the flow
        adaptive_restart=False,
        verbose=True,
        arrows=True,
        eps=0.01,  # regularization parameter
        N=250,  # number of particles
        max_time=100,  # max time horizon
        d=5,  # dimension of the particles
        subdiv=1000,  # number of subdivisions of [0, max_time]
        sigma=1  # Gaussian kernel parameter
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
    nu = np.ones(d)  # target mean
    Q = np.eye(d) * 5  # target covariance
    Q_inv = np.linalg.inv(Q)  # inverse target covariance matrix

    def target_score(x):
        return Q_inv @ (x - nu)

    target_type = 'Gaussian'
    target_mean = nu
    target_cov = Q

    def target_density(x, y):
        point = np.dstack((x, y))
        return sp.stats.multivariate_normal.pdf(point, mean=nu, cov=Q)

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

    # target_score = U3_grad
    # target_density = U3_density
    # target_type = 'squiggly2'
    # target_mean = np.zeros(2)
    # target_cov = np.eye(2) + 1/4*np.ones((2, 2))

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
    make_folder(folder_name+'/accelerated')
    make_folder(folder_name+'/non-accelerated')
    # restart_counter is used to compute the acceleration parameter.
    # It will be reset to 1 when a restart is triggered.
    restart_counter = 1

    # non-accelerated particles
    X_non = X.copy()

    Xs = np.zeros((subdiv, N, d))
    Xs_non = np.zeros((subdiv, N, d))
    X_prev = X.copy()
    L = []
    for k in tqdm(range(subdiv)):
        if d == 2 and plot and not k % 25:
            plot_particles(X, Y, k, folder_name, target_density, 'k',
                           'accelerated', arrows)
            plot_particles(X_non, None, f'non_acc_{k}', folder_name,
                           target_density, 'b', 'non-accelerated', arrows)

        X_old = X_prev.copy()
        X_prev = X.copy()
        Xs[k, :, :] = X
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

    create_gif(folder_name+'/accelerated',
               f'{folder_name}/{folder_name}_acc.gif')
    create_gif(folder_name+'/non-accelerated',
               f'{folder_name}/{folder_name}_non_acc.gif')

    # plotting quantities along the flow
    # if target_type == 'Gaussian':
    plt.plot(means, label='deviation from mean')
    plt.plot(covs, label='deviation from cov')
    plt.plot(KLs, label='KL between empirical cov matrix and target cov')
    plt.title(f'N ={N}, d = {2}, eps = {eps}, tau = {tau}')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{folder_name}/{folder_name}_loss.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    plot_paths(Xs, folder_name)
    plot_paths(Xs_non, folder_name, 'non-acc')


acc_Stein_Particle_Flow()
