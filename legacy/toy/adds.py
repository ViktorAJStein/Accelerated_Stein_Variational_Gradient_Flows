# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 10:35:45 2025

Additional functions

@author: Viktor Stein
"""
import numpy as np
import os
from PIL import Image


# KL divergence of N(mu, A) to N(nu, B)
def KL(A, B, B_inv, mu, nu):
    _, logdet_A = np.linalg.slogdet(A)
    _, logdet_B = np.linalg.slogdet(B)
    first = np.trace(B_inv @ A)
    second = logdet_B - logdet_A
    third = (nu - mu).T @ B_inv @ (nu - mu)
    d = A.shape[0]
    return 0.5 * (first - d + second + third)


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
