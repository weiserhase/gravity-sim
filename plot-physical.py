import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

import physical as phys


class GravityPlot(object):
    def __init__(self, engine: phys.SimulationEngine):
        self.engine = engine

        self.fig, self.axs = plt.subplots(2, figsize=(10, 30))
        plt.grid()
        self.animated = anim.FuncAnimation(
            self.fig, self.update, interval=5, init_func=self.setup_plot, blit=True)
        print("01")

    def setup_plot(self):

        x, y = np.meshgrid(
            np.arange(200), np.arange(200))

        # self.field = self.axs[0].quiver(x, y, self.velocity[:, :, 0],
        #                                 self.velocity[:, :, 1])
        col, pos, labels = self.split_data(self.engine.objects)
        x, y = pos[:, :2]
        self.scatter = self.axs[1].scatter(
            x, y, s=labels, c=col)
        self.axs[1].set_aspect('equal', adjustable='box')
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        plt.xlim([0, 300])
        plt.ylim([0, 300])
        # self.axs[1].set_ylim(xmin-4*(xmax-xmin), xmax+4*(xmax-xmin))

        # self.axs[1].set_xlim(ymin-4*(ymax-ymin), ymax+4*(ymax-ymin))
        return self.scatter,

    def split_data(self, data: dict[int, phys.Sphere]):
        labels = np.zeros((len(data)))
        positions = np.zeros((len(data), 3))
        col = []
        c_map = {0: "red", 1: "green"}
        for key, obj in data.items():
            positions[key] = obj.position
            labels[key] = obj.mass
            col.append(c_map[obj.identifier])
        return col, positions, labels * 20/labels.max()

    # def vector_

    def update(self, i):
        # timings = [time.time()]
        for i in range(100):
            data = self.engine.step()
        # timings[-1] = time.time() - timings[-1]
        # timings.append(time.time())
        col, pos, s = self.split_data(data)
        # print(pos, s*100)
        # x, y = pos[:, :2]
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        print(pos, s, "label and position")
        print(x, y, "Xy")
        # plt.scatter(x, y, s=s)
        # plt.show()
        # timings[-1] = time.time() - timings[-1]
        # timings.append(time.time())

        self.scatter.set_offsets(np.c_[x, y])
        # timings[-1] = time.time() - timings[-1]
        # timings.append(time.time())
        self.scatter.set_sizes(
            np.interp(s, [s.min(), s.max()], [5, 10])
        )
        # Update Plot size
        # self.axs[1].relim()
        # self.axs[1].autoscale_view(True, True, True)

        # xmin = x.min()
        # xmax = x.max()
        # ymin = y.min()
        # ymax = y.max()
        # self.axs[1].set_xlim(xmin-0.1*(xmax-xmin), xmax+0.1*(xmax-xmin))
        # self.axs[1].set_ylim(ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin))

        # input()
        # plt.show()
        # timings[-1] = time.time() - timings[-1]
        # print(timings)
        return self.scatter,


def split_data(data: dict[int, phys.Sphere]):
    labels = np.zeros((len(data)))
    positions = np.zeros((len(data), 3))
    col = []
    c_map = {0: "red", 1: "green"}
    print(data)
    for key, obj in data.items():
        positions[key] = obj.position
        labels[key] = obj.mass
        col.append(c_map[obj.identifier])
    return col, positions, labels


def manual_plot(data):
    col, pos, labels = split_data(data)
    print(pos)
    x = pos[:, 0]
    y = pos[:, 1]
    # np.append(x, 0)
    # np.append(y, 0)
    # np.append(labels, 200)
    # print(x, y, y, labels)

    plt.scatter(x, y, s=transform(labels), c=col)
    plt.xlim([0, 300])
    plt.ylim([0, 300])
    plt.show()


def test_plot():

    # obj = {}
    # masses = [10**8, 10**8]
    # for i in range(2):
    #     pos = np.zeros((3))
    #     pos[:2] = np.random.randint(0, 50, 2)
    #     vel = np.zeros((3))
    #     vel[:2] = np.random.rand(2) * 1/8
    #     obj[i] = phys.Sphere(i, masses[i], pos, vel)
    #     print(obj[i].position, "Position origin")
    # print("--------------------------------")

    obj = {}
    masses = [10**8, 10**3, 10**4]
    pos = np.array([[150, 170, 0], [150, 40, 0], [100, 100, 0]])
    vel = np.array(([0, 0, 0], [0.03, 0, 0], [0.0, 0, 0]))
    for i in range(2):
        obj[i] = phys.Sphere(i, masses[i], pos[i], vel[i])
        obj[i].print()
    # grid[0, 10] =

    engine = phys.SimulationEngine(obj)
    # for i in range(100000):
    #     m = engine.step()
    #     if (i % 1000 == 0):
    #         print(engine.objects)
    #         for key, j in engine.objects.items():
    #             print(j.velocity, "Velocity", key)
    #         manual_plot(m)

    plot = GravityPlot(engine)
    plt.show()


if __name__ == '__main__':
    test_plot()
