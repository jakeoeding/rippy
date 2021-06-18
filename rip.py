import matplotlib.pyplot as plt
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

def compute_gradients(img, edge_x, edge_y):
    gaussian = generate_gaussian_kernel()
    kernel_x = signal.convolve2d(gaussian, edge_x, 'same')
    kernel_y = signal.convolve2d(gaussian, edge_y, 'same')
    gradient_x = signal.convolve2d(img, kernel_x, 'same')
    gradient_y = signal.convolve2d(img, kernel_y, 'same')
    return gradient_x, gradient_y

def compute_gradient_magnitude(gradient_x, gradient_y):
    return np.sqrt(gradient_x ** 2 + gradient_y ** 2)


if __name__ == '__main__':
    img_path = 'img/broadway_tower.jpg'
    img = Image.open(img_path)
    img_rgb = np.array(img)
    img_gray = np.array(img.convert('L'))
    gx, gy = compute_gradients(img_gray, SOBEL_X, SOBEL_Y)
    energy_matrix = compute_gradient_magnitude(gx, gy)
    plt.imshow(energy_matrix)
    plt.show()