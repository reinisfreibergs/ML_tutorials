import numpy as np


class Character(object):
    def __init__(self):
        self.__geometry = []
        self.__angle = 0.0
        self.__speed = 0.1
        self.__pos = np.array([0, 0])
        self.__dir = np.array([0, 1])
        self.__color = 'r'
        self.C = np.identity(3)
        self.R = np.identity(3)
        self.T = np.identity(3)


    def draw(self):
        x_values = []
        y_values = []
        for vec2d in self.geometry:
            x_values.append(vec2d[0])
            y_values.append(vec2d[1])

        plt.plot(x_values, y_values, c=self.color)

    def generate_geometry(self):
        pass

class Player(Character):
    def __init__(self):
        super().__init__()

    def generate_geometry(self):
        self.geometry = np.array([
            [-1, 0],
            [1, 0],
            [0, 1],
            [-1, 0]
        ])


class Asteroid(Character):
    def __init__(self):
        super().__init__()

    def generate_geometry(self):
        self.geometry = []
