from dataclasses import dataclass

import numpy as np


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1-vec2)


def potential_gradient(sphere1: "Sphere", vector: np.ndarray, G: float):
    res = np.zeros_like(sphere1.position, dtype=float)

    if sphere1.position.shape != vector.shape:
        raise Exception("Sphere and Vector dont have the same shape")
    # x-Component
    for i in range(sphere1.position.shape[0]):
        indices = np.delete(np.arange(vector.shape[0]), i)
        # print((2 * sphere1.mass * G), (2*(sphere1.position[i] - vector[i])) + np.sum(
        #     (sphere1.position[indices]-vector[indices])**2))
        # print((2 * sphere1.mass * G) /
        #       ((2*(sphere1.position[i] - vector[i]) +
        # np.sum((sphere1.position[indices]-vector[indices])**2))))

        res[i] = ((sphere1.position[i] - vector[i]) * sphere1.mass * G) /\
            np.sum((sphere1.position-vector)**2)**2
        # print(res[i])

    # print(res)
    return res


def acc_grid(sphere1: "Sphere", vector: np.ndarray, G: float) -> np.ndarray:
    if sphere1.position.shape != vector.shape:
        raise Exception("Sphere Positions dont have the same shape")

    magnitude = (sphere1.mass * G) / \
        euclidean_distance(vector, sphere1.position)**2

    direction = (sphere1.position - vector)

    return direction * magnitude


def sphere_vec_acc(sphere1: "Sphere", vector: np.ndarray, G):
    if sphere1.position.shape != vector.shape:
        raise Exception("Sphere Positions dont have the same shape")

    magnitude = (sphere1.mass * G) / \
        euclidean_distance(vector, sphere1.position)**2

    direction = (sphere1.position - vector)

    return direction * magnitude

def acc(sphere1: "Sphere", sphere2: "Sphere", G: float):
    return sphere_vec_acc(sphere1, sphere2.position,G)


@ dataclass
class Sphere:
    def __init__(self, identifier: int, mass: float, position: np.ndarray, velocity: np.ndarray):
        self.identifier = identifier
        self.mass = mass
        self.position = position
        self.velocity = velocity

    def change_velocity(self, vel_change: np.ndarray):
        # print(vel_change, "velocity setter")
        self.velocity = vel_change + self.velocity
        return self

    def update_position(self, position: np.ndarray):
        # print(self.position, position, "Position setter")

        self.position = position + self.position
        return self

    def update(self, acceleration: np.ndarray, time_delta: float):
        # print(f"-----------------{self.identifier}---------------")
        # print(acceleration, "acceleration: ", time_delta)
        # print("Velocity: ", self.velocity)
        # print("Position: ", self.position)

        self.change_velocity(acceleration*time_delta)
        self.update_position(self.velocity*time_delta)
        # print("Velocity: ", self.velocity)
        # print("Position: ", self.position)
        # print("-------------------END-----------------")
        return self

    def print(self):
        print(self.__dict__)


class SimulationEngine:
    def __init__(self, objects: dict[int, Sphere]) -> None:
        self.objects: dict[int, Sphere] = objects
        self.G = 6.67*10**(-11)
        self.last_dst = np.infty
        self.range = self.plot_range()
        self.range =(0,0,200,200)
    def step(self) -> dict[int, Sphere]:
        time_delta = 1
        obj_mesh = np.array(np.meshgrid(
            list(self.objects.values()), list(self.objects.values())))\
            .ravel("F").reshape(-1, 2)
        # print(obj_mesh)
        grad_map = {}  # np.zeros((len(self.objects), 3))
        for el1, el2 in obj_mesh:
            # print(el1.identifier, el2.identifier)

            if el1.identifier == el2.identifier:
                continue
            grad_map[el2.identifier] = acc(el1, el2, self.G)
            # grad_map[el2.identifier] = potential_gradient(
            #     el1, el2.position, self.G)
            # print(potential_gradient(el1, el2.position, self.G), np.linalg.norm(
            #     potential_gradient(el1, el2.position, self.G)))

        new_obj_map = {}
        for identifier, acceleration in grad_map.items():
            obj = self.objects[identifier]
            # print(obj.position, "Position")
            prev_pos = obj.position
            obj.update(acceleration, time_delta)
            # print(np.linalg.norm(obj.position-prev_pos), obj.identifier)
            # print(obj.position, "Position2")

            new_obj_map[identifier] = obj

        self.objects = new_obj_map
        # print(new_obj_map, "New objects")
        curr_dst = (np.linalg.norm(
            self.objects[0].position - self.objects[1].position))
        # if curr_dst > self.last_dst:
        #     # raise Exception("Distance Increasing")
        self.last_dst = curr_dst
        return new_obj_map
    def plot_range(self):
        pos = np.array([obj.position for obj in self.objects.values()])
        return pos[:,0].min(), pos[:,1].min(), pos[:,0].max(), pos[:,1].max()

    def all_vectors(self, granularity=1):
        minx,miny,maxx,maxy = self.range
        plotxmin, plotymin, plotxmax, plotymax = minx * 0.75, miny * 0.75, maxx * 1.25, maxy * 1.25
        linx = np.linspace(plotxmin, plotxmax, int((plotxmax - plotxmin) // granularity))
        liny = np.linspace(plotymin, plotymax, int((plotymax - plotymin) // granularity))
        coords = np.array(np.meshgrid(linx, liny)).ravel("F").reshape(-1,2)
        vectors = np.zeros((coords.shape[0],2,3))
        for i, (x,y) in enumerate(coords):
            vectors[i,0,:] =np.array([x,y,0])
            acc_vect = np.zeros((3))
            for obj in self.objects.values():
                acc_vect = acc_vect + sphere_vec_acc(obj, np.array([x,y,0]), self.G)
            vectors[i,1,:] = acc_vect
        return vectors



def main():
    obj = {}
    masses = [10**6, 10**4]
    pos = np.array([[0, 5, 0], [0, 1, 0]])
    vel = np.array(([0, 0, 0], [0, 0, 0]))
    for i in range(2):
        obj[i] = Sphere(i, masses[i], pos[i], vel[i])
    # print(obj)
    engine = SimulationEngine(obj)
    for i in range(10000):
        sim_map = engine.step()
        if i % 100 == 0:
            for key, value in sim_map.items():
                print(key, value.print())
            # map(lambda k, x: x.print(), sim_map.items())


if __name__ == '__main__':
    main()
