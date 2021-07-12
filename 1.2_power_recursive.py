def pow_recursive(a, b):
    if b == 1:
        power = a
    else:
        power = a * pow_recursive(a, b - 1)
    return power


print(pow_recursive(3, 5))
