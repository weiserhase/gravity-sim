from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


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
