import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from csv import reader

train_file = '0.csv'

def raw_to_pixel(l):
    '''Convert the raw coordinates to pixel coordinates.'''
    return [int(x)/5 for x in l]

def raw_cartesian_to_polar_angles(l):
    '''Convert the cartesian coordinates to polar coordinates.'''
    x_red, y_red, x_green, y_green, x_blue, y_blue = raw_to_pixel(l)

    angle_green_red = np.arctan((y_green-y_red)/(x_green-x_red+1e-5))
    angle_blue_green = np.arctan((y_blue-y_green)/(x_blue-x_green+1e-5))

    return [np.degrees(angle_green_red), np.degrees(angle_blue_green)]

with open('0.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    coordinates = list(csv_reader)

#(x_red, y_red, x_green, y_green, x_blue, y_blue)
angles_1 = []
angles_2 = []

for row in coordinates:
    angle = raw_cartesian_to_polar_angles(row)
    angles_1.append(angle[0])
    angles_2.append(angle[1])

t = np.linspace(0,len(coordinates)/400, len(coordinates))
plt.figure(figsize=(10,10))
plt.plot(t, angles_1)
# plt.xlim([0,40])
plt.show()



