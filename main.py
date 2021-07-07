def power(a, b):  # doesn't work for non-whole powers
    result = 1
    try:
        for multipliers in range(abs(b)):
            if b >= 0:
                result = result * a
            else:
                result = result / a
    except:
        result = -1
    return result


def e_x(x):
    ex = 1 + x + power(x, 2)/2 + power(x, 3)/6 + power(x, 4)/24 + power(x, 5)/120 + power(x, 6)/720
    return ex

print(e_x(1))


def natural_log(x):
    natural = 0
    for i in range(1,50):
        natural += power(-1,(i+1)) * power((x-1),i)/i
    #natural = (x-1) - pow((x-1), 2)/2 + pow((x-1), 3)/3 - pow((x-1), 4)/4
    return natural

print(natural_log(1.5))


def pow_non_whole(a, b): #for any float power satisfying 0<b<2
    e = e_x(1)
    power = b*natural_log(a)
    pow = e_x(power)
    return pow

print (pow_non_whole(0.7, 0.7))
