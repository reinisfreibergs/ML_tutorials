import numpy as np

# print(a[0, 0], a[0, 1])  # prints definite elements
# print(a[:, 0])  # prints row 0
# print(a[:, 1])  # prints row 1
# print(a.shape)  # returns rows and columns in tuple
# print(a.shape[0])  # prints rows

A = np.array([
    [1, 3, 6],
    [5, 2, 8]
])

B = np.array([
    [1, 3],
    [5, 2],
    [6, 9]
])
C = np.array([1, 2, 3])
D = np.array([1, 2])


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


#print(dot(A, B))
#print(dot(B, A))
print(dot(C, B))
print(np.dot(C, B))
print(np.dot(B, D))
print(dot(B, D))

# print(dot(B, C))


