import numpy as np
import time

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 10)
plt.ion()


def rotation_mat(degrees):
    theta = np.radians(degrees)
    cos = np.cos(theta)
    sin = np.sin(theta)
    R = np.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]
    ])
    return R


def translation_mat(dx, dy):
    T = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])
    return T


def scale_mat(sx, sy):
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0 ,1]
    ])
    return S


def scew_mat(x, y):
    W = np.array([
        [1, np.tan(x), 0],
        [np.tan(y), 1, 0],
        [0, 0 ,1]
    ])
    return W


def dot(X, Y):

    is_transposed = False

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    if X.shape[1] != Y.shape[0]:
        is_transposed = True
        Y = np.transpose(Y)

    X_rows = X.shape[0]
    Y_rows = Y.shape[0]
    Y_columns = Y.shape[1]

    product = np.zeros((X_rows, Y_columns))
    for i in range(X_rows):
        for j in range(Y_columns):
            for k in range(Y_rows):
                product[i, j] += X[i, k] * Y[k, j]

    if is_transposed:
        product = np.transpose(product)

    if product.shape[0] == 1:
        product = product.flatten()


    return product


def vec2d_to_vec3d(vec2):
    I  = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    vec3 = dot(I, vec2) + np.array([0, 0, 1])
    return vec3


def vec3d_to_vec2d(vec3):
    I  = np.array([
        [1, 0, 0],
        [0, 1, 0],
    ])
    vec2 = dot(I, vec3)
    return vec2

# vec2 = np.array([1.0, 0])
# vec3 = vec2d_to_vec3d(vec2)
# vec2 = vec3d_to_vec2d(vec3)
# print(vec3)
# print(vec2)
# exit()


class Character(object):
    def __init__(self):
        super().__init__()
        self.__angle = np.random.random() * np.pi

        self.geometry = []
        self.color = 'k'

        self.C = np.identity(3)
        self.R = np.identity(3)
        self.S = np.identity(3)
        self.T = np.identity(3)

        self.pos = np.array([0.0, 0.0])
        self.dir_init = np.array([0.0, 1.0])
        self.dir = np.array(self.dir_init)
        self.speed = 0.25


        self.generate_geometry()

    def set_angle(self, angle):
        self.__angle = angle  # encapsulation
        self.R = rotation_mat(angle)

        vec3d = vec2d_to_vec3d(self.dir_init)
        vec3d = dot(self.R, vec3d)
        self.dir = vec3d_to_vec2d(vec3d)

        self.__update_transformation()

    def get_angle(self):
        return self.__angle


    def update_movement(self, dt):
        self.pos += self.dir * self.speed * dt
        self.__update_transformation()


    def __update_transformation(self):
        self.T = translation_mat(self.pos[0], self.pos[1])
        T_centered = translation_mat(dx= 0, dy= -0.75)

        self.C = T_centered
        self.C = dot(self.R, self.C)
        self.C = dot(self.T, self.C)


    def set_position(self, pos):
        self.pos = pos
        self.T = translation_mat(self.pos[0], self.pos[1])

    def get_position(self):
        return self.pos

    def draw(self):
        x_values = []
        y_values = []

        ship_scaled = scale_mat(sx=0.2, sy=1.5)
        self.C = dot(self.C, ship_scaled)

        for vec2d in self.geometry:

            vec3d = vec2d_to_vec3d(vec2d)
            vec3d = dot(self.C, vec3d)
            vec2d = vec3d_to_vec2d(vec3d)

            x_values.append(vec2d[0])
            # print(vec2d)
            y_values.append(vec2d[1])

        plt.plot(x_values, y_values, c=self.color)


class Asteroid(Character):
    def __init__(self):
        super().__init__()

        self.colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        self.color = self.colours[np.random.randint(0,7)]

        self.pos = -10 * np.random.random((2,)) +5
        self.speed = np.random.random() * 0.5 + 0.1
        self.set_angle(np.random.random() * 360)

    def generate_geometry(self):
        self.geometry = []
        step = 2 * np.pi/20
        radius = 0.8 * np.random.random() + 0.5
        theta = 0.0

        while theta < 2 * np.pi:
            a = 0.3 * np.random.random(2) + 0.5

            self.geometry.append(np.array([
                a[0] * np.cos(theta) * radius,
                a[1] * np.sin(theta) * radius
            ]))
            theta += step

        self.geometry.append(np.array(self.geometry[0]))

    def draw(self):

        margins = [10, -10]
        x_values = []
        y_values = []

        asteroid_scaled = scew_mat(x=0.1, y=0.5)
        self.C = dot(self.C, asteroid_scaled)

        for vec2d in self.geometry:

            vec3d = vec2d_to_vec3d(vec2d)
            vec3d = dot(self.C, vec3d)
            vec2d = vec3d_to_vec2d(vec3d)

            if round(vec2d[0],1) in margins:
                # self.set_angle(self.get_angle() + 1)
                self.speed *= -1
            if round(vec2d[1], 1) in margins:
                # self.set_angle(self.get_angle() - 1)
                self.speed *= -1

            x_values.append(vec2d[0])
            y_values.append(vec2d[1])

        plt.plot(x_values, y_values, c=self.color)


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

class Bullet(Character):
    def __init__(self):
        super().__init__()

        self.pos = np.array(player.get_position())
        self.T = translation_mat(self.pos[0], self.pos[1])
        self.speed = 3
        self.set_angle(player.get_angle())

    def generate_geometry(self):
        self.geometry = np.array([
            [-1, 0],
            [1, 0],
        ])


characters = []
for i in range(10):
    characters.append(Asteroid())
characters.append(Player())
player = characters[-1]

is_running = True


def press(event):
    global is_running, player
    print('press', event.key)
    if event.key == 'escape':
        is_running = False  # quits app
    elif event.key == 'right':
        player.set_angle(player.get_angle() - 5)
    elif event.key == 'left':
        player.set_angle(player.get_angle() + 5)
    elif event.key == 'down':
        player.set_position(player.get_position()-1)
    elif event.key == 'up':
        player.set_position(player.get_position()+1)
    elif event.key == 'x':
        characters.append(Bullet())


fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)


dt = 0.1
while is_running:
    plt.clf()

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    start = time.time()
    for character in characters:  # polymorphism
        character.update_movement(dt)
        character.draw()

    dt = 0.3 + time.time() - start
    print(dt)
    plt.draw()
    plt.pause(1e-2)
