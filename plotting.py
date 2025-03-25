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
    xmin, xmax, ymin, ymax = target.lims
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
    T = target.density(X, Y)
    if target.name == "GMM_scale_density":
        plt.contour(X, Y, T, levels=np.arange(25)/25, colors='black', alpha=.2)
    else:
        plt.contour(X, Y, T, levels=7, colors='black', alpha=.2)
    plt.title(f'Iteration {label}')
    plt.grid('True')
    plt.savefig(f'{folder_name}/{acc}/{folder_name}_{label}.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_paths(k, X, non, under, over, MALA, target, folder_name):
    plot_paths(k, X, folder_name, target)
    plot_paths(k, non, folder_name, target, 'SVGD')
    plot_paths(k, under, folder_name, target, 'ULD')
    plot_paths(k, over, folder_name, target, 'ULA')
    plot_paths(k, MALA, folder_name, target, 'MALA')


def plot_paths(k, Xs, folder_name, target, add=''):
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
    for i in range(N):
        # Extract the trajectory for particle i.
        x = Xs[:k, i, 0]
        y = Xs[:k, i, 1]
        # Mark the starting point with a circle (blue).
        plt.plot(x[0], y[0], color=cmap(norm(0)), marker='o', ms=5)
        # Mark the endpoint with a square (red).
        plt.plot(x[-1], y[-1], color=cmap(norm(k-1)), marker='s', ms=5)
    # Perform a kernel density estimate on the data:
    xmin, xmax, ymin, ymax = target.lims
    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # plot contour lines of target density
    T = target.density(X, Y)
    if target.name == "GMM_scale_density":
        plt.contour(X, Y, T, levels=np.arange(25)/25, colors='black', alpha=.2)
    else:
        plt.contour(X, Y, T, levels=7, colors='black', alpha=.2)

    # plt.title(f'Particle Trajectories {add}')
    # plt.grid(True)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.savefig(f'{folder_name}/{folder_name}_{add}_paths.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def plotKL(k, acc, non, over, under, MALA, lnZ, folder_name):
    plt.plot(np.arange(k), acc[:k]+lnZ, label='ASVGD')
    plt.plot(np.arange(k), non[:k]+lnZ, label='SVGD')
    plt.plot(np.arange(k), over[:k]+lnZ, label='ULA')
    plt.plot(np.arange(k), under[:k]+lnZ, label='ULD')
    plt.plot(np.arange(k), MALA[:k]+lnZ, label='MALA')
    plt.legend()
    plt.yscale('symlog')
    plt.xlabel('Iterations')
    plt.grid()
    plt.ylabel('Monte-Carlo approximation of KL')
    plt.savefig(f'{folder_name}/{folder_name}_KL.png',
                dpi=300, bbox_inches='tight')
    plt.show()
