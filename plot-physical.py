import matplotlib as mpl
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

import physical as phys
from objects import Sphere


class GravityPlot(object):
    def __init__(self, engine: phys.SimulationEngine):
        self.engine = engine

        self.fig = plt.figure()
        self.axs = self.fig.subplot_mosaic(
            [["potential", "potential", "field", "field"], ["potential", "potential", "gravity", "gravity"]], height_ratios=[5, 5], width_ratios=[2, 2, 2, 2])
        self.fig.tight_layout(pad=2)

        self.animated = anim.FuncAnimation(
            self.fig, self.update, interval=5, init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        # Vector fields
        vector, pot, mesh = self.engine.all_vectors(20, 5)
        coords = vector[:, 0, :]
        vectors = vector[:, 1, :]
        x = coords[:, 0]
        y = coords[:, 1]

        self.field = self.axs["field"].quiver(x, y, vectors[:, 0],
                                              vectors[:,  1])

        self.axs["field"].set_aspect('equal', adjustable='box')

        # Absolute Position
        x, y = np.meshgrid(
            np.arange(200), np.arange(200))

        col, pos, labels = self.split_data(self.engine.objects)
        x = pos[:, 0]
        y = pos[:, 1]
        ss = self.axs["gravity"].get_subplotspec()
        self.axs["gravity"].remove()
        self.axs["gravity"] = self.fig.add_subplot(
            ss,
        )
        # self.fig.axs = self.fig.add_axes((0,40,0,40))
        self.scatter = self.axs["gravity"].scatter(
            x, y, s=labels, c=col)
        self.axs["gravity"].set_aspect('equal', adjustable='box')
        self.axs["gravity"].set_xlim(0, 400)
        self.axs["gravity"].set_ylim(0, 400)
        self.axs["gravity"].grid()
        norm = plt.Normalize(pot.min(), pot.max())
        colors = mpl.cm.viridis(norm(pot))
        ss = self.axs["potential"].get_subplotspec()
        self.axs["potential"].remove()
        self.axs["potential"] = self.fig.add_subplot(ss, projection="3d")

        X, Y, Z = mesh[0], mesh[1], pot.reshape(-1, mesh[0].shape[0])
        print(X.shape, Y.shape, Z.shape)
        rcount, ccount = 30, 30
        # colors=colors)  # , [pot.min(), pot.max()], [0, 10] cmap=mpl.cm.magma
        self.potential = self.axs["potential"].plot_surface(
            X, Y, np.log10(Z),  lw=0.1, rcount=rcount, ccount=ccount,
            cmap=mpl.cm.viridis, alpha=0.6, edgecolor="grey")

        # self.axs["potential"].contourf(
        #     X, Y, Z, zdir='z', offset=-9.5, cmap='coolwarm')
        # self.axs["potential"].contourf(
        #     X, Y, Z, zdir='x', offset=-300, cmap='coolwarm')
        # self.axs["potential"].contourf(
        #     X, Y, Z, zdir='y', offset=-300, cmap='coolwarm')
        self.axs["potential"].set(
            xlabel='X', ylabel='Y', zlabel='Z')
        # self.potential = self.axs["potential"].plot_surface(
        # mesh[0], mesh[1], np.log10(pot.reshape(-1, mesh[0].shape[0])), cmap=mpl.cm.magma)

        xmin, ymin, xmax, ymax = self.engine.plot_range()
        # plt.xlim([xmin*-1, xmax*3])
        # plt.ylim([ymin*-1, ymax*3])
        # self.axs = self.fig.add_subplot(2, 1, 3, projection='3d')

        # plt.setp(self.axs[0][1], x_lim=(0, 400), y_lim=(0, 400))
        # self.axs[1].set_ylim(xmin-4*(xmax-xmin), xmax+4*(xmax-xmin))

        # self.axs[1].set_xlim(ymin-4*(ymax-ymin), ymax+4*(ymax-ymin))
        return self.scatter, self.field, self.potential

    def split_data(self, data: dict[int, phys.Sphere]):
        labels = np.zeros((len(data)))
        positions = np.zeros((len(data), 3))
        col = []
        c_map = {0: "red", 1: "green", 2: "blue", 3: "yellow"}
        for key, obj in data.items():
            positions[key] = obj.position
            labels[key] = obj.mass
            col.append(c_map[obj.identifier])
        return col, positions, labels * 20/labels.max()

    # def vector_

    def update(self, i):
        # timings = [time.time()]
        for i in range(25):
            data = self.engine.step()
        # timings[-1] = time.time() - timings[-1]
        # timings.append(time.time())
        col, pos, s = self.split_data(data)
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        self.scatter.set_offsets(np.c_[x, y])
        self.scatter.set_sizes(
            np.interp(s, [s.min(), s.max()], [5, 10])
        )

        # self.field.remove()
        vector_fields, pot, mesh = self.engine.all_vectors(20, 100)
        # print(vector_fields.shape)
        # coords = vector_fields[:,0,:]
        vectors = vector_fields[:, 1, :]
        # vectors[np.linalg.norm(vectors) > np.percentile(np.linalg.norm(
        #     vectors), 10)] = [0.1, 0.1, 0.1]
        # x = coords[:,0]
        # y = coords[:,1]
        # self.field.remove()
        self.field.set_UVC(vectors[:, 0], vectors[:, 1])
        # self.axs["gravity"].relim()
        # self.axs["gravity"].autoscale_view()

        # if self.axs["potential"].collections is not None:

        #     self.axs["potential"].collections.remove(self.potential)
        # self.potential = self.axs["potential"].plot_wireframe(mesh[0], mesh[1], np.log10(
        #     pot.reshape(-1, mesh[0].shape[0])), rstride=5, cstride=5)
        # self.axs["potential"].collections.remove(self.potential)
        # self.potential = self.axs["potential"].plot_wireframe(mesh[0], mesh[1], np.log10(
        #     pot.reshape(-1, mesh[0].shape[0])), rstride=5, cstride=5)
        # self.axs["potential"].clear()
        # self.potential = self.axs["potential"].plot_surface(
        # mesh[0], mesh[1], pot.reshape(-1, mesh[0].shape[0]))
        # self.potential = self.axs["potential"].set_data(
        #     pot.reshape(-1, mesh[0].shape[0]))
        # self.field = self.axs[0].quiver(x, y, vectors[ :, 0],
        #                                 vectors[:,  1])
        # Update Plot size
        # self.axs[1].relim()
        # self.axs[1].autoscale_view(True, True, True)
        # xmin, ymin, xmax, ymax = self.engine.plot_range()
        # # xmin = x.min()
        # # xmax = x.max()
        # # ymin = y.min()
        # # ymax = y.max()
        # self.axs[1].set_xlim(xmin-0.1*(xmax-xmin), xmax+0.1*(xmax-xmin))
        # self.axs[1].set_ylim(ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin))

        return self.scatter, self.field, self.potential


def split_data(data: dict[int, phys.Sphere]):
    labels = np.zeros((len(data)))
    positions = np.zeros((len(data), 3))
    col = []
    c_map = {0: "red", 1: "green", 2: "blue",
             3: "yellow", 4: "black", 5: "purple"}
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
    masses = [10**8, 10**5, 10**6]
    pos = np.array([[150, 150, 0], [150, 50, 0], [100, 100, 0]])
    vel = np.array(([0, 0, 0], [0.08, 0, 0], [0.04, 0, 0]))
    for i in range(3):
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
