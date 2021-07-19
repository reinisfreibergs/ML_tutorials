import sklearn.datasets._california_housing
import numpy as np
import matplotlib.pyplot as plt

X, Y = sklearn.datasets.fetch_california_housing(return_X_y = True)

# print(X[:5])
# print(Y[:5])
dala = X[:5]
daly = Y[:5]


def normalize_value(value, minimum, maximum):
    new_value = 2 * ((value - minimum)/(maximum - minimum) - 0.5)
    return new_value


def min_max_finder(data):
    column_data = []
    minmax = []
    if np.ndim(data) == 1:
        data = np.transpose(np.expand_dims(data, axis = 0))
    # print(np.shape(data))
    for k in range(np.shape(data)[1]):
        for i in range(np.shape(data)[0]):
            column_data.append(data[i][k])
        minimum = min(column_data)
        maximum = max(column_data)
        minmax.append([minimum, maximum])
        # print(minmax)
        column_data = []
    return minmax

def normalize(data):

    if np.ndim(data) == 1:
        data = np.transpose(np.expand_dims(data, axis = 0))

    dataset = np.zeros_like(data)
    min_max = min_max_finder(data)

    for k in range(np.shape(data)[1]):
        for i in range(np.shape(data)[0]):
            dataset[i][k] += normalize_value(data[i][k], min_max[k][0], min_max[k][1])

    return dataset

print(normalize(dala))
print(normalize(daly))

