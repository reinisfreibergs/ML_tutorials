def pow_recursive(a, b):
    if b == 1:
        return a
    else:
        power = a * pow(a, b - 1)
    return power


print(pow_recursive(3, 5))
