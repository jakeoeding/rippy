import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from PIL import Image
from scipy import signal


SOBEL_X = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

SOBEL_Y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])


def generate_gaussian_kernel(size=5, sigma=1):
    kernel1d = signal.gaussian(size, sigma).reshape(size, 1)
    return np.outer(kernel1d, kernel1d)

@jit
def rgb2gray(img_rgb):
    return img_rgb[:, :, 0] * 0.2989 + img_rgb[:, :, 1] * 0.5870 + img_rgb[:, :, 2] * 0.1140

def compute_gradients(img_rgb):
    img_gray = rgb2gray(img_rgb)
    gaussian = generate_gaussian_kernel()
    gaus_grad_x = signal.fftconvolve(gaussian, SOBEL_X, 'same')
    gaus_grad_y = signal.fftconvolve(gaussian, SOBEL_Y, 'same')
    img_grad_x = signal.fftconvolve(img_gray, gaus_grad_x, 'same')
    img_grad_y = signal.fftconvolve(img_gray, gaus_grad_y, 'same')
    return img_grad_x, img_grad_y

def compute_energy(img_rgb):
    gx, gy = compute_gradients(img_rgb)
    return np.sqrt(gx ** 2 + gy ** 2)

def accumulate_energy(energy):
    rows, columns = energy.shape

    middle_ixs = np.arange(columns)
    left_ixs = np.clip(middle_ixs - 1, 0, columns)
    right_ixs = np.clip(middle_ixs + 1, 0, columns - 1)

    cumulative_energy = np.zeros((rows, columns))
    cumulative_energy[0] = energy[0]

    for i in range(1, rows):
        prior_row = cumulative_energy[i - 1]
        min_ancestor = np.minimum(prior_row[left_ixs], prior_row[middle_ixs])
        min_ancestor = np.minimum(min_ancestor, prior_row[right_ixs])
        cumulative_energy[i] = energy[i] + min_ancestor

    return cumulative_energy

@jit
def trace_seam(cumulative_energy):
    rows, columns = cumulative_energy.shape

    seam = np.zeros(rows, dtype=np.int32)
    seam[-1] = np.argmin(cumulative_energy[-1])

    for i in range(rows - 2, -1, -1):
        row = cumulative_energy[i]
        middle = seam[i + 1]
        left = max(0, middle - 1)
        right = min(columns - 1, middle + 1)
        options = np.array([row[left], row[middle], row[right]])
        direction = np.argmin(options) - 1
        seam[i] = max(middle + direction, 0)

    return seam

def find_vertical_seam(energy):
    cumulative_energy = accumulate_energy(energy)
    return trace_seam(cumulative_energy)

@jit
def carve_vertical_seam(current_rgb, seam):
    original_shape = current_rgb.shape
    new_rgb = np.zeros((original_shape[0], original_shape[1] - 1, original_shape[2]))

    for i in range(original_shape[0]):
        new_rgb[i, :, 0] = np.delete(current_rgb[i, :, 0], seam[i])
        new_rgb[i, :, 1] = np.delete(current_rgb[i, :, 1], seam[i])
        new_rgb[i, :, 2] = np.delete(current_rgb[i, :, 2], seam[i])

    return new_rgb

def carve_vertical(img, num_seams):
    img_rgb = np.array(img)
    for _ in range(num_seams):
        energy = compute_energy(img_rgb)
        seam = find_vertical_seam(energy)
        img_rgb = carve_vertical_seam(img_rgb, seam)
    return Image.fromarray(img_rgb.astype(np.uint8), 'RGB')

def carve_horizontal(img, num_seams):
    horz_img = img.transpose(Image.ROTATE_270)
    img_out = carve_vertical(horz_img, num_seams)
    return img_out.transpose(Image.ROTATE_90)

def carve(img, num_vert=0, num_horz=0):
    if num_vert:
        img = carve_vertical(img, num_vert)
    if num_horz:
        img = carve_horizontal(img, num_horz)
    return img


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-img', help='Input image path', required=True)
    parser.add_argument('-out', help='Output image path', required=True)
    parser.add_argument('-vert', help='Number of vertical seams to remove', type=int, default=0)
    parser.add_argument('-horz', help='Number of horizontal seams to remove', type=int, default=0)

    args = parser.parse_args()
    path_in, path_out = args.img, args.out
    vert, horz = args.vert, args.horz

    img = Image.open(path_in)
    img_out = carve(img, vert, horz)
    img_out.save(path_out)