import asyncio
import functools
import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

np.set_printoptions(linewidth=np.infty, threshold=sys.maxsize)


def fft_convolve2d(x, y):
    """
    2D convolution, using FFT
    """
    fourier_x = np.fft.fft2(x)
    fourier_y = np.fft.fft2(np.flipud(np.fliplr(y)))
    m, n = fourier_x.shape
    cc = np.real(np.fft.ifft2(fourier_x*fourier_y))
    cc = np.roll(cc, - int(m / 2) + 1, axis=0)
    cc = np.roll(cc, - int(n / 2) + 1, axis=1)
    return cc


def random_grid(size):
    return np.random.rand(size, size)


def euclidean_distance(vec1, vec2, scale):
    return 1 / (np.linalg.norm(vec1*1/scale-vec2*1/scale) + 0.0000001)**2

    # return math.sqrt(np.sum(vectors**2))


def generate_coeff_matrix(test_size, scale):

    coords = np.array(np.meshgrid(np.arange(test_size), np.arange(test_size))
                      ).ravel("F").reshape(-1, 2)
    coeff_matrix = np.zeros((test_size, test_size), dtype=float)

    p1 = np.array([test_size//2, test_size//2])
    for coord_pair in coords:
        p2 = coord_pair
        coeff_matrix[tuple(p2)] = euclidean_distance(p1, p2, scale)
    coeff_matrix[tuple(p1)] = 0
    return coeff_matrix


def partial_grid(coords, size, grid):
    padded_grid = np.pad(grid, size)
    # print(padded_grid)
    x, y = coords
    # print(x, y)
    # print(x-size//2+size)
    # print(y-size//2+size, y+size//2+1+size)
    # print(x-size//2+size, x-size//2+1+size)
    return padded_grid[x-size//2+size: x+size//2+1+size, y-size//2+size: y+size//2+1+size]


def generate_coords(shape):
    return np.array(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))).ravel("F").reshape(-1, 2)


def direction_vectors(shape):
    coords = generate_coords((shape, shape))
    return np.apply_along_axis(
        lambda x: x-np.array([shape//2, shape//2]), 1, coords).reshape(shape, shape, 2)


def in_bounds(shape, coords):
    res = np.array(coords)
    if coords[0] < 0:
        res[0] = 0
    elif coords[0] >= shape[0]:
        res[0] = shape[0] - 1

    if coords[1] < 0:
        res[1] = 0
    elif coords[1] >= shape[1]:
        res[1] = shape[1] - 1
    return res


class Simulation:

    def __init__(self, grid, velocity=None, neigh_size=5, time_delta: float = 1, scale=50):
        self.grid = grid
        self.neigh_size = neigh_size
        self.time_delta = time_delta
        self.scale = scale

        if velocity is None or velocity.shape[:2] != self.grid.shape:
            self.velocity = np.zeros(
                (self.grid.shape[0], self.grid.shape[1], 2))
        else:
            self.velocity = velocity

        # Generate Masks
        self.coeff_matrix = generate_coeff_matrix(neigh_size, scale)
        self.shifted_coords = direction_vectors(neigh_size)

    def step(self):
        grid_copy = np.copy(self.grid)
        gravity_const = 6.67*10**(-11)
        func = functools.partial(
            partial_grid, size=self.neigh_size, grid=np.copy(grid_copy))
        time_delta = self.time_delta
        new_velocity = np.zeros_like(self.velocity)
        new_grid = np.zeros_like(grid_copy)
        # vector_space = np.zeros((grid_copy[0], grid_copy[1], int(self.neigh_size), int(self.neigh_size)))
        for val in generate_coords(np.copy(grid_copy).shape):
            x, y = val
            if self.grid[x, y] == 0:
                continue
            # print("--------------------------------")
            # print(x, y)
            # print("----------------END---------------")
            partial = func((x, y))
            # print(partial)
            try:
                gravity_magnitude = self.coeff_matrix * \
                    gravity_const*partial * grid_copy[x, y]
                # print(partial, grid_copy[x, y], self.coeff_matrix)
            except Exception as e:
                print(e)
                print(partial, "Partial MAtrix print")

            gravity_magnitude[self.neigh_size//2, self.neigh_size//2] = 0
            acc_vector = np.zeros((2))
            # print(generate_coords((self.neigh_size, self.neigh_size)))
            for px, py in generate_coords((self.neigh_size, self.neigh_size)):
                # print(gravity_magnitude[px, py],
                #       self.shifted_coords[px, py], (px, py), (x, y))
                acc_vector += gravity_magnitude[px,
                                                py] * self.shifted_coords[px, py]

                # if np.linalg.norm(gravity_magnitude[px,
                #                                     py] * self.shifted_coords[px, py]) > 0:

                #     print(acc_vector,
                #           self.shifted_coords[px, py]+np.array([x, y]),
                #           (x, y),
                #           gravity_magnitude[px, py],
                #           self.shifted_coords[px, py])

            pos_change = self.velocity[x, y] * time_delta

            def round_np(arr):
                res = np.zeros_like(arr)
                for i, el in enumerate(arr):
                    if (el > 0):
                        res[i] = np.floor(el)
                    else:
                        res[i] = np.ceil(el)
                return res.astype(int)
            new_pos = round_np(pos_change)+np.array([x, y])
            # print(new_pos, (x, y))
            new_pos = in_bounds(grid_copy.shape, new_pos)
            nx, ny = new_pos
            # print(acc_vector)
            # if np.linalg.norm(self.velocity) > 0:
            # print(self.velocity[x, y], acc_vector *
            #       time_delta, (x, y), "Vel")
            new_velocity[nx, ny] = new_velocity[nx, ny] + self.velocity[x, y] +\
                (acc_vector * time_delta)
            new_grid[nx, ny] = new_grid[x, y] + grid_copy[x, y]
            # print(in_bounds(grid_copy.shape, new_pos))

        self.grid = new_grid
        self.velocity = new_velocity
        return new_grid, new_velocity


class GravityPlot(object):
    def __init__(self, velocity, grid, engine: Simulation):
        self.engine = engine
        self.grid = grid
        self.velocity = velocity

        self.fig, self.axs = plt.subplots(2)
        self.animated = FuncAnimation(
            self.fig, self.update, interval=100, init_func=self.setup_plot, blit=True)

    def setup_plot(self):

        x, y = np.meshgrid(
            np.arange(self.grid.shape[0]), np.arange(self.grid.shape[1]))

        # self.field = self.axs[0].quiver(x, y, self.velocity[:, :, 0],
        #                                 self.velocity[:, :, 1])
        self.scatter = self.axs[1].scatter(
            x, y, s=self.grid*(20/self.grid.max()))
        return self.scatter,

    def split_data(self, data):
        coords = np.where(data > 0)
        labels = data[coords]
        scaled_labels = labels * 10/data.max()
        return coords[0], coords[1], scaled_labels

    def update(self, i):
        timings = [time.time()]

        data = self.engine.step()[0]
        timings[-1] = time.time() - timings[-1]
        timings.append(time.time())
        x, y, s = self.split_data(data)
        timings[-1] = time.time() - timings[-1]
        timings.append(time.time())
        self.scatter.set_offsets((x, y))
        timings[-1] = time.time() - timings[-1]
        timings.append(time.time())
        self.scatter.set_sizes(
            s
        )
        input()
        timings[-1] = time.time() - timings[-1]
        print(timings)
        return self.scatter,


def test_plot():
    size = 100

    grid = np.zeros((size, size), dtype=float)
    grid[30, 30] = 2000000
    grid[50, 50] = 10**6
    # grid[0, 10] =

    velocities = np.zeros((size, size, 2))
    velocities[6, 6] = [0, 1.00]
    # print(grid)
    engine = Simulation(grid, velocities, size*2+1, 1)
    plot = GravityPlot(velocities, grid, engine)
    plt.show()


def main():
    size = 50

    grid = np.zeros((size, size), dtype=float)
    grid[20, 20] = 2000000
    grid[50, 50] = 10**7
    # grid[0, 10] =

    velocities = np.zeros((size, size, 2))
    velocities[20, 20] = [0, 5.00]
    print(grid)
    engine = Simulation(grid, velocities, size*2+1, 0.5)
    total_weight = [grid.sum()]
    import os

    # Plots

    for i in range(3000):
        engine.step()
        total_weight.append(engine.grid.sum())
        os.system("cls")
        # print(engine.grid.astype(int))
        scatter.set_sizes(
            (engine.grid*(8/engine.grid.max())).flatten().ravel())
        plt.show()
        # plot_grid(engine.grid, engine.velocity)
        # print(engine.velocity)

        # print(engine.velocity)
        # print(total_weight[-1] - total_weight[-2])
        input()


if __name__ == '__main__':
    # part = partial_grid((1, 1), 3, np.arange(16).reshape(4, 4))
    # print(part)
    # pass
    test_plot()
