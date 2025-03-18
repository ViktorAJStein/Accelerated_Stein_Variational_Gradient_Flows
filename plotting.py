# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 10:36:58 2025

@author: Viktor Stein
"""
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl


def plot_particles(particles, velocities, label, folder_name,
                   target, c, acc, arrows, KDE=False):
    m1, m2 = particles[:, 0], particles[:, 1]
    xmin = -8  # m1.min() - 1.0
    xmax = 8  # m1.max() + 1.0
    ymin = -8  # m2.min() - 1.0
    ymax = 8  # m2.max() + 1.0
    # Perform a kernel density estimate on the data:
    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
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
                   angles='xy', scale_units='xy', scale=1, alpha=.2)
    ax.plot(m1, m2, '.', markersize=2, color=c)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_aspect('equal', adjustable='box')
    # plot contour lines of target density
    T = target(X, Y)
    if target.__name__ == "GMM_scale_density":
        plt.contour(X, Y, T, levels=np.arange(25)/25, colors='black', alpha=.2)
    else:
        plt.contour(X, Y, T, levels=7, colors='black', alpha=.2)
    plt.title(f'Iteration {label}')
    plt.grid('True')
    plt.savefig(f'{folder_name}/{acc}/{folder_name}_{label}.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_paths(k, Xs, folder_name, add=''):
    N = Xs.shape[1]
    plt.figure(figsize=(10, 8))
    # Set up colormap and normalization from blue (start) to red (end)
    cmap = plt.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=k-1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    for i in range(N):
        # Extract the trajectory for particle i.
        x = Xs[:k, i, 0]
        y = Xs[:k, i, 1]
        # Create line segments to be colored individually.
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Create a LineCollection with a color gradient along the segments.
        lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.2)
        # Set the array for the colormap; here, we use the time index.
        lc.set_array(np.linspace(0, k-1, len(segments)))
        lc.set_linewidth(2)
        ax.add_collection(lc)
        # Mark the starting point with a circle (blue).
        plt.plot(x[0], y[0], color=cmap(norm(0)), marker='o', ms=5)
        # Mark the endpoint with a square (red).
        plt.plot(x[-1], y[-1], color=cmap(norm(k-1)), marker='s', ms=5)

    plt.title(f'Particle Trajectories {add} \n'
              + 'Starting points are circles, endpoints are squares')
    plt.grid(True)
    plt.xlim([-7, 7])
    plt.ylim([-5, 8])
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
    plt.plot(np.arange(k), acc[:k]+lnZ, label='accelerated SVGD')
    plt.plot(np.arange(k), non[:k]+lnZ, label='non-accelerated SVGD')
    plt.plot(np.arange(k), over[:k]+lnZ, label='overdamped Langevin')
    plt.plot(np.arange(k), under[:k]+lnZ, label='underdamped Langevin')
    plt.plot(np.arange(k), MALA[:k]+lnZ, label='MALA')
    plt.legend()
    plt.yscale('symlog')
    plt.grid()
    plt.title('Monte-Carlo approximation of KL')
    plt.savefig(f'{folder_name}/{folder_name}_KL.png',
                dpi=300, bbox_inches='tight')
    plt.show()
