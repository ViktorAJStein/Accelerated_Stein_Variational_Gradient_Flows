# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 10:36:58 2025

@author: Viktor Stein
"""
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

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
            values = particles.T
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


def plot_paths(k, Xs, folder_name, add=''):
    N = Xs.shape[1]
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, N)]
    plt.figure(figsize=(10, 8))
    for i in range(N):
        # Plot the trajectory with transparency (alpha).
        plt.plot(Xs[:k, i, 0], Xs[:k, i, 1], c=colors[i], alpha=0.2)
        # Mark the starting point with a circle marker.
        plt.plot(Xs[0, i, 0], Xs[0, i, 1], marker='o', c=colors[i], ms=5)
        # Mark the endpoint with a square marker.
        plt.plot(Xs[k-1, i, 0], Xs[k-1, i, 1], marker='s', c=colors[i], ms=5)
    plt.title(f'Particle Trajectories {add} \n'
              + folder_name + '\n'
              + ' Starting points are circles, endpoints are squares')
    plt.grid(True)
    plt.xlim([-8, 8])
    plt.ylim([-5, 5])
    plt.savefig(f'{folder_name}/{folder_name}_{add}_paths.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_paths(k, X, non, under, over, MALA, folder_name):
    plot_paths(k, X, folder_name)
    plot_paths(k, non, folder_name, 'non-acc')
    plot_paths(k, under, folder_name, 'underdamped')
    plot_paths(k, over, folder_name, 'overdamped')
    plot_paths(k, MALA, folder_name, 'MALA')


def plotKL(k, acc, non, over, under, MALA, lnZ, folder_name):
    # Define the window size for the moving average
    window_size = 10
    window = np.ones(window_size) / window_size

    plt.plot(np.arange(k), acc[:k]+lnZ, label='accelerated')
    plt.plot(np.arange(k), non[:k]+lnZ, label='non-accelerated', alpha=.2)
    plt.plot(np.arange(k), np.convolve(non[:k], window, mode='same')+lnZ, label='non-accelerated (smoothed)')
    plt.plot(np.arange(k), over[:k]+lnZ, label='overdamped Langvin')
    plt.plot(np.arange(k), under[:k]+lnZ, label='underdamped Langevin')
    plt.plot(np.arange(k), MALA[:k]+lnZ, label='MALA')
    plt.legend()
    plt.grid()
    plt.title('Monte-Carlo approximation of KL')
    plt.savefig(f'{folder_name}/{folder_name}_KL.png',
                dpi=300, bbox_inches='tight')
    plt.show()

