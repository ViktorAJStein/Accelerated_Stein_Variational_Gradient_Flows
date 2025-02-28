# -*- coding: utf-8 -*-
"""
Particle algorithm to approximate Stein gradient flow with Gaussian kernel
K(x, y) = exp(-1/(2 sigma^2) * || x - y ||_2^2) w.r.t. the KL divergence

@author: Viktor Stein
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import scipy as sp
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
                duration=70,  # in milliseconds
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


def plot_particles(particles, label, folder_name, target, KDE=False):
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
    ax.plot(m1, m2, 'k.', markersize=2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_aspect('equal', adjustable='box')
    # plot contour lines of target density
    T = target(X, Y)
    plt.contour(X, Y, T, levels=5, colors='black', alpha=.2)
    plt.title(f'Iteration {label}')
    plt.grid('True')
    plt.savefig(f'{folder_name}/{folder_name}_{label}.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_paths(Xs, folder_name):
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
    plt.title('Particle Trajectories \n'
              + folder_name + '\n'
              + ' Starting points are circles, endpoints are squares')
    plt.grid(True)
    plt.savefig(f'{folder_name}/{folder_name}_paths.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def U2_density(x, y):
    return np.exp(- 25/8 * (y - np.sin(np.pi/2*x))**2)


def U2_grad(x):
    factor = U2_density(x[0], x[1]) * (x[1] - np.sin(np.pi / 2 * x[0]))
    return - factor * np.array([25/8 * np.pi * np.cos(np.pi / 2 * x[0]), -25/4])


def U3_density(x, y):
    first = np.exp(-1/2*((y - np.sin(np.pi / 2 * x))/(.35))**2)
    w2 = 3*np.exp(-1/2*((x-1)/(.6))**2)
    second = np.exp(-1/2*((y - np.sin(np.pi / 2 * x) + w2)/(.35))**2)
    return first + second


def U3_grad(x):
    term1 = np.exp(-200/49 * (x[1] - np.sin((np.pi * x[0]) / 2))**2)
    term2 = np.exp(-25/18 * (x[0] - 1)**2)
    term3 = np.exp(-200/49 * (3 * term2 - np.sin((np.pi * x[0]) / 2) + x[1])**2)
    A = 200 * np.pi * np.cos((np.pi * x[0]) / 2) * term1 * (x[1] - np.sin((np.pi * x[0]) / 2))
    B = -25/3 * term2 * (x[0] - 1) - 0.5 * np.pi * np.cos((np.pi * x[0]) / 2)
    C = 3 * term2 - np.sin((np.pi * x[0]) / 2) + x[1]
    num1 = -(A - 2 * B * C * term3)
    den1 = 49 * (term3 + term1)
    first_expr = num1 / den1
    num2 = (-400/49 * C * term3 - 400/49 * term1 * (x[1] - np.sin((np.pi * x[0]) / 2)))
    den2 = term3 + term1
    second_expr = - (num2 / den2)
    return np.array([first_expr, second_expr])


def U4_density(x, y):
    first = np.exp(-1/2*((y - np.sin(np.pi / 2 * x))/(.4))**2)
    w3 = 3/(1 + np.exp(10/3*(1-x)))
    second = np.exp(-1/2*((y - np.sin(np.pi / 2 * x) + w3)/(.35))**2)
    return first + second


def U4_grad(x):
    exp_a = np.exp(-25/8 * (x[1] - np.sin((np.pi * x[0]) / 2))**2)
    exp_b = np.exp((10 * (1 - x[0])) / 3)
    common_expr = 3/(exp_b + 1) - np.sin((np.pi * x[0]) / 2) + x[1]
    exp_c = np.exp(-200/49 * common_expr**2)
    denom = exp_c + exp_a
    num1 = -(
        (25/8) * np.pi * np.cos((np.pi * x[0]) / 2) * exp_a * (x[1] - np.sin((np.pi * x[0]) / 2))
        - (400/49) * (
            (10 * exp_b) / ((exp_b + 1)**2) - 0.5 * np.pi * np.cos((np.pi * x[0]) / 2)
        ) * (
            3/(exp_b + 1) - np.sin((np.pi * x[0]) / 2) + x[1]
        ) * exp_c
    )
    first_component = num1 / denom
    num2 = -(
         - (400/49) * common_expr * exp_c
         - (25/4) * exp_a * (x[1] - np.sin((np.pi * x[0]) / 2))
    )
    second_component = num2 / denom
    return np.array([first_component, second_component])


def bimodal_density(x, y):
    first = np.exp(-2*(np.sqrt(x**2 + y**2) - 3)**2)
    second = np.exp(-2*(x-3)**2) + np.exp(-2*(x+3)**2)
    return first * second


def bimodal_grad(x):
    nx = np.linalg.norm(x)
    first = 4 * x * (1 - 3 / nx)
    second = np.array([
        8*(np.exp(24*x[0]) * (x[0] - 3) + x[0] + 3) / (np.exp(24*x[0]) + 1),
        0])
    return first + second


def bimodal2_density(x, y):
    first = np.exp(-2*(np.sqrt(x**2 + y**2) - 3)**2)
    second = np.exp(-2*(y-3)**2) + np.exp(-2*(y+3)**2)
    return first * second


def gauss_mix_density(x, y, a1=1/2, a2=1/2):
    first = np.exp(-1/2*(x-a1)**2 - 1/2*(y - a2)**2)
    return 1/(4*np.pi) * (first + np.exp(-1/2*(x+a1)**2 - 1/2*(y + a2)**2))


def gauss_mix_grad(x, a=np.array([1, 1])):
    # gradient of the potential of N(a, Id) + N(-a, Id)
    return x - a + 2 * a / (1 + np.exp(2 * np.dot(x, a)))


def acc_Stein_Particle_Flow(
        plot=True,  # decide whether to plot the particles along the flow
        adaptive_restart=True,
        verbose=True,
        eps=.01,  # regularization parameter
        N=500,  # number of particles
        max_time=200,  # max time horizon
        d=2,  # dimension of the particles
        subdiv=200,  # number of subdivisions of [0, max_time]
        Q=np.array([[3, -2], [-2, 3]]),  # target covariance
        ):
    init_mean = np.zeros(d)  # initial mean
    init_cov = np.ones(d) @ np.ones(d) + np.eye(d)  # initial covariance
    nu = np.ones(d)  # target mean
    # X, _ = make_moons(N, noise=.1, random_state=42)  # initial particles
    # X = sampling(N, bimodal2_density)
    # prior_name = 'bimodal_2'
    # uniform time step size
    tau = max_time / subdiv
    X = np.random.multivariate_normal(init_mean, init_cov, size=N)
    prior_name = 'Gaussian'
    # initial velocities
    Y = np.zeros((N, d))
    # inverse covariance matrix
    Q_inv = np.linalg.inv(Q)

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

    # target_score = U3_grad
    # target_density = U3_density
    # target_type = 'squiggly2'
    # target_mean = np.zeros(2)
    # target_cov = np.eye(2) + 1/4*np.ones((2, 2))

    target_score = U4_grad
    target_density = U4_density
    target_type = 'squiggly3'
    target_mean = np.zeros(2)
    target_cov = np.eye(2) + 1/4*np.ones((2, 2))

    # target_score = bimodal_grad
    # target_density = bimodal_density
    # target_type = 'non-Gaussian'
    # target_mean = np.zeros(2)
    # target_cov = np.array([[1.43, 0], [0, 9.125]])
    # Gaussian kernel parameter
    sigma = .1

    # initialize quantities tracked along the flow
    means = np.zeros(subdiv)
    covs = np.zeros(subdiv)
    KLs = np.zeros(subdiv)

    # print initial data
    folder_name = (f'N={N},d={d},mu_0={init_mean.flatten()},'
                   + f'Sigma_0={init_cov.flatten()},eps={eps},'
                   + f'max_time={max_time},subdiv={subdiv},tau={tau},'
                   + f'{target_type}_target_restart={adaptive_restart},'
                   + f'sigma={sigma},prior={prior_name}'
                   )
    make_folder(folder_name)
    # restart_counter is used to compute the acceleration parameter.
    # It will be reset to 1 when a restart is triggered.
    restart_counter = 1

    Xs = np.zeros((subdiv, N, d))
    X_prev = X.copy()
    L = []
    for k in tqdm(range(subdiv)):
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
        try:
            V = np.linalg.pinv(K + N*eps*np.eye(N)) @ Y
        except Exception:
            break
        # plot particles
        if d == 2:
            empirical_mean = np.mean(X, axis=0)
            empirical_cov = np.cov(X, rowvar=False)
            means[k] = np.linalg.norm(empirical_mean - target_mean)
            covs[k] = np.linalg.norm(empirical_cov - target_cov)
        if d == 1:
            empirical_cov = empirical_cov * np.eye(1)
        if d <= 2:
            KLs[k] = KL(empirical_cov, Q, Q_inv, empirical_mean, nu)
        # two quantities for adaptive restart
        norm_diff_current = np.linalg.norm(X - X_prev)  # ||x_{k+1} - x_k||
        norm_diff_prev = np.linalg.norm(X_prev - X_old)  # ||x_k - x_{k-1}||

        if d == 2 and plot and not k % 5:
            plot_particles(X, k, folder_name, target_density)
        # adaptive restart
        if adaptive_restart and k > 0:
            if target_type == 'Gaussian' and KLs[k] - KLs[k - 1] > 0:
                if verbose:
                    print(f'No KL-descent at iteration {k}, restarting momentum')
                restart_counter = 1
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
        if KLs[k] < 1e-7:
            print('Functional values is below 1e-7, stopping iteration.')
            break
    plt.plot(L, label='\| W - Id \|')
    plt.legend()
    plt.show()
    print('Final empirical mean and covariance matrix:'
          + f' {empirical_mean}, {empirical_cov}')

    create_gif(folder_name, f'{folder_name}/{folder_name}.gif')

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


acc_Stein_Particle_Flow()
