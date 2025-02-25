# -*- coding: utf-8 -*-
"""
Particle algorithm to approximate Stein gradient flow with bilinear kernel
K(x, y) = x^T A y + 1 w.r.t. the KL divergence with Gaussian target

@author: Viktor Stein
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from scipy.stats import gaussian_kde
from tqdm import tqdm
import os
from PIL import Image

# Set random seed for reproducibility
np.random.seed(42)


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


def plot_particles(particles, label, folder_name):
    # plt.figure(figsize=(8, 6))
    # plt.scatter(particles[:, 0], particles[:, 1],
    #             c='b', marker='o', edgecolors='k', alpha=.2)
    # plt.title(label)
    # plt.grid(True)
    # plt.axis('equal')
    # plt.xlim(-8, 8)
    # plt.ylim(-5, 5)
    m1, m2 = particles[:, 0], particles[:, 1]
    xmin = -8  # m1.min() - 1.0
    xmax = 8  # m1.max() + 1.0
    ymin = -8  # m2.min() - 1.0
    ymax = 8  # m2.max() + 1.0
    # Perform a kernel density estimate on the data:
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    # Plot the results:
    fig, ax = plt.subplots()
    ax = plt.gca()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
              extent=[xmin, xmax, ymin, ymax])
    ax.plot(m1, m2, 'k.', markersize=2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_aspect('equal', adjustable='box')
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
    plt.show()


def acc_Stein_Particle_Flow(
        plot=True,  # decide whether to plot the particles along the flow
        eps=0,  # regularization parameter
        N=500,  # number of particles
        max_time=10,  # max time horizon
        d=2,  # dimension of the particles
        subdiv=1000,  # number of subdivisions of [0, max_time]
        Q=np.array([[3, -2], [-2, 3]]),  # target covariance
        ):
    init_mean = np.ones(d)  # initial mean
    init_cov = np.ones(d) @ np.ones(d) + np.eye(d)  # initial covariance
    nu = np.zeros(d)  # target mean
    # X, _ = make_moons(N, noise=.1, random_state=42)  # initial particles

    # uniform time step size
    tau = max_time / subdiv
    X = np.random.multivariate_normal(init_mean, init_cov, size=N)
    # initial velocities
    Y = np.zeros((N, d))
    # inverse covariance matrix
    Q_inv = np.linalg.inv(Q)
    # kernel parameter matrix
    A = 1/2 * Q_inv

    # initialize quantities tracked along the flow
    means = np.zeros(subdiv)
    covs = np.zeros(subdiv)
    KLs = np.zeros(subdiv)

    # print initial data
    folder_name = (f'N={N},d={d},mu_0={init_mean.flatten()},'
                   + f'Sigma_0={init_cov.flatten()},A={A.flatten()},eps={eps}'
                   + f'max_time={max_time},subdiv={subdiv},tau={tau}'
                   )
    make_folder(folder_name)
    # restart_counter is used to compute the acceleration parameter.
    # It will be reset to 1 when a restart is triggered.
    restart_counter = 1

    Xs = np.zeros((subdiv, N, d))
    for k in tqdm(range(subdiv)):
        Xs[k, :, :] = X
        X += tau * Y
        # kernel matrix
        K = X @ A @ X.T + np.ones((N, N))
        try:
            V = N * np.linalg.pinv(K + N*eps*np.eye(N)) @ Y
        except:
            break
        # plot particles
        empirical_mean = np.mean(X, axis=0)
        empirical_cov = np.cov(X, rowvar=False)
        means[k] = np.linalg.norm(empirical_mean - nu)
        covs[k] = np.linalg.norm(empirical_cov - Q)
        if d == 1:
            empirical_cov = empirical_cov * np.eye(1)
        KLs[k] = KL(empirical_cov, Q, Q_inv, empirical_mean, nu)
        if d == 2 and plot and not k % 20:
            plot_particles(X, k, folder_name)
        # adaptive restart
        # alternative: if || x_{k + 1} - x_k || < || x_k - x_{k - 1} ||
        if KLs[k] - KLs[k - 1] > 0 and k > 0:
            print(f'No KL-descent at iteration {k}, restarting momentum')
            restart_counter = 1
        else:
            restart_counter += 1
        # acceleration parameter
        alpha_k = (restart_counter - 1) / (restart_counter + 2)
        # update velocities
        Y = (1 - tau*alpha_k) * Y + tau * (X @ A - K @ (X - nu) @ Q_inv / N)
        Y += tau / (N * N) * np.trace(V @ V.T @ K) * X @ A

        # early stopping for efficiency
        if KLs[k] < 1e-7:
            print('Functional values is below 1e-7, stopping iteration.')
            break

    print('Final empirical mean and covariance matrix:'
          + f' {empirical_mean}, {empirical_cov}')

    create_gif(folder_name, f'{folder_name}/{folder_name}.gif')

    # plotting quantities along the flow
    plt.plot(means, label='deviation from mean')
    plt.plot(covs, label='deviation from cov')
    plt.plot(KLs, label='functional value')
    plt.title(f'N ={N}, d = {2}, eps = {eps}, A = {A}, tau = {tau}')
    plt.legend()
    plt.yscale('log')
    plt.show()

    plot_paths(Xs, folder_name)


acc_Stein_Particle_Flow()