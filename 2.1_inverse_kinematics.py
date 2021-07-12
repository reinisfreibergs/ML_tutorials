import torch
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
plt.ion()

target_point = np.array([-3.0, 0])
anchor_point = np.array([0, 0])

is_running = True
def button_press_event(event):
    global target_point
    target_point = np.array([event.xdata, event.ydata])

def press(event):
    global is_running
    print('press', event.key)
    if event.key == 'escape':
        is_running = False # quits app

fig, _ = plt.subplots()
fig.canvas.mpl_connect('button_press_event', button_press_event)
fig.canvas.mpl_connect('key_press_event', press)

length_joint = 2.0
theta_1 = np.deg2rad(-10)
theta_2 = np.deg2rad(-10)
theta_3 = np.deg2rad(-10)

def rotation(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    R = np.array([
        [cos, -sin],
        [sin, cos]
    ])
    return R

def d_rotation(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    R = np.array([
        [-sin, -cos],
        [cos, -sin]
    ])
    return R

def translation_mat(dx, dy):
    T = np.array([
        [1, dx],
        [0, dy]
    ])
    return T

loss = 0
step = 1e-2
unit = np.array([0.0, 1.0]) * length_joint
while is_running:
    plt.clf()
    plt.title(f'loss: {round(loss, 4)} theta_1: {round(np.rad2deg(theta_1))} theta_2: {round(np.rad2deg(theta_2))}')

    joints = []

    R1 = rotation(theta_1)
    d_R1 = d_rotation(theta_1)

    R2 = rotation(theta_2)
    d_R2 = d_rotation(theta_2)

    R3 = rotation(theta_3)
    d_R3 = d_rotation(theta_3)

    joints.append(anchor_point)

    first_point = np.dot(R1, unit)
    joints.append(first_point)

    # T = translation_mat(first_point[0],first_point[1])
    # affine_transformation = np.dot(T, R2)
    # second_point = np.dot(affine_transformation, unit)
    # joints.append(second_point)

    second_point = R1 @ (unit + R2 @ unit)
    joints.append(second_point)

    third_point = R1 @ (unit + R2 @ unit + R3 @ unit)
    joints.append(third_point)

    loss = np.sum((target_point - third_point) ** 2)

    d_theta_1 = np.sum(d_R1 @ unit * -2*(target_point - third_point))
    theta_1 -= d_theta_1 * step

    d_theta_2 = np.sum(R1 @ d_R2 @ unit * -2*(target_point - third_point))
    theta_2 -= d_theta_2 * step

    d_theta_3 = np.sum(R1 @ R2 @ d_R3 @ unit * -2*(target_point - third_point))
    theta_3 -= d_theta_3 * step


    np_joints = np.array(joints)

    if len(np_joints):
        plt.plot(np_joints[:, 0], np_joints[:, 1])
    plt.scatter(target_point[0], target_point[1], s=50, c='r')

    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    plt.draw()
    plt.pause(1e-3)
    #break
# input('end')

