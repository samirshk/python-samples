from math import *

def say_hi(name):
    print("hello " + name)

print(len("Hello World".lower()))

j = "no\nno"
m = True

say_hi(j.upper())
print(m)
print(j[0])

my_num = -10

print(str(abs(my_num)) + " is good")
print(pow(my_num, 2))
print(ceil(4.5))

name = input("what is your name? ")
age = float(input("what is your age? "))



print("hello " + name.upper() + "! you are: " + str(age))


color = input("color? ")
plural_noun = input ("plural noun? ")
celebrity = input("celebrity? ")

print("Roses are " + color)
print(plural_noun +" are blue")
print("I love " + celebrity)

my_list = ["samir", "shaikh"]
my_copy = my_list.copy()

coord_str = input("enter coord")

x = int(coord_str[0])
y = int(coord_str[1])

coord = (x, y)

print(coord)
print(coord[0] == coord[1])
if coord[0] < coord[1]:
    say_hi("left bound coord")
elif coord[0] == coord[1]:
    say_hi("on the line")
else:
    say_hi("right bound coord")
