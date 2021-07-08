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


def factorial(x):
    if x == 0:
        fact = 1
    else:
        fact = x * factorial(x-1)

    return fact


def e_x(x):
    ex = 0
    for i in range(50):
        ex += power(x, i)/factorial(i)
        #ex = 1 + x + power(x, 2)/2 + power(x, 3)/6 + power(x, 4)/24 + power(x, 5)/120 + power(x, 6)/720
    return ex

print(e_x(1))


def natural_log_range_two(x):
    natural = 0
    for i in range(1,50):
        natural += power(-1,(i+1)) * power((x-1),i)/i
    #natural = (x-1) - pow((x-1), 2)/2 + pow((x-1), 3)/3 - pow((x-1), 4)/4
    return natural

print(natural_log_range_two(1.5))


def closest_power_of_two(x):#x = m*2^p for any real number, where 0.5<m<=1
    product = 2
    count = 1

    while product<abs(x):
        product *=2
        count+=1

    final_product = x/product

    return final_product,count


def natural_log(x): #x = m*2^p
    ln2 = 0.693147180559945
    m = closest_power_of_two(x)[0]
    p = closest_power_of_two(x)[1]
    log = natural_log_range_two(m) + p * ln2

    return log


def pow_non_whole(a, b): #for any float power satisfying 0<b<2
    power = b*natural_log(a)
    pow = e_x(power)

    return pow

print (pow_non_whole(0.7, 0.7))
print (pow_non_whole(2, 2))
print(pow_non_whole(0.123, 0.123))
