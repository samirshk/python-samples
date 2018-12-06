print (3**3)

def raise_to_power(base_num, pow_num):
    result = 1
    for i in range(pow_num):
        result *= base_num
    return result

print(raise_to_power(3,3))
print(raise_to_power(3,4))