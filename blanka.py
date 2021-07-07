import numpy as np

# product = np.zeros(shape=(3))
# print(product)
# print(product[0])
# print(product.shape[0])

I  = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])


def a(x):
    x+=1
    return x
print (a(I))
