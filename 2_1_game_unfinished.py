import numpy as np

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
    S = np.identity(3)
    # TODO
    return S


def dot(X, Y):

    X_rows = X.shape[0]
    Y_rows = Y.shape[0]

    if X.ndim == 1:

        Y_columns = Y.shape[1]
        product = np.zeros(Y_columns)
        for i in range(X_rows):
            for j in range(Y_columns):
                product[j] += X[i] * Y[i, j]

    elif Y.ndim ==1:

        product = np.zeros(X_rows)
        for i in range(X_rows):
            for k in range(Y_rows):
                product[i] += X[i, k] * Y[k]

    else:
        Y_columns = Y.shape[1]
        product = np.zeros((X_rows, Y_columns))
        for i in range(X_rows):
            for j in range(Y_columns):
                for k in range(Y_rows):
                    product[i, j] += X[i, k] * Y[k, j]

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
        self.color = 'r'

        self.C = np.identity(3)
        self.R = np.identity(3)
        self.S = np.identity(3)
        self.T = np.identity(3)

        self.pos = np.array([5.0, 5.0])
        self.dir_init = np.array([0.0, 1.0])
        self.dir = np.array(self.dir_init)
        self.speed = 0.1

        self.T = translation_mat(self.pos[0], self.pos[1])


        self.generate_geometry()

    def set_angle(self, angle):
        self.__angle = angle  # encapsulation
        self.R = rotation_mat(self.__angle)

    def get_angle(self):
        return self.__angle

    def set_position(self, pos):
        self.pos = pos
        self.T = translation_mat(self.pos[0], self.pos[1])

    def get_position(self):
        return self.pos




    def draw(self):
        x_values = []
        y_values = []

        self.C = dot(self.T, self.R)

        for vec2d in self.geometry:

            vec3d = vec2d_to_vec3d(vec2d)

            vec3d = dot(self.C, vec3d)

            vec2d = vec3d_to_vec2d(vec3d)

            x_values.append(vec2d[0])
            y_values.append(vec2d[1])

        plt.plot(x_values, y_values, c=self.color)


class Asteroid(Character):
    def __init__(self):
        super().__init__()

    def generate_geometry(self):
        self.geometry = []


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


characters = []
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


fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)

while is_running:
    plt.clf()

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    for character in characters:  # polymorphism
        character.draw()

    plt.draw()
    plt.pause(1e-2)
