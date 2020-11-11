# !! Python code cells
#Exercise 1:

print('excerise 1')

# !! Using Python as a calculator
#  Exercise 2:
print(2 + 3)
print(2 - 3)
print(2 * 3)
print(2 / 3)
print(2 ** 3)

print(38 / 5)
print(38 // 5)
print(38 % 5)

# !! Importing modules for more functionality
# eg
import numpy as np
print(np.sin(np.pi))
# Exercise 3:
import numpy as np
A = np.sqrt(2)*np.cos(2*np.pi/5)
print (A)

# !! Variable assignment
# eg
a = 0.5
b = 6
c = a + b
print(b)
print(c)
# Exercise 4:
x = 1
y = x
x = 2
print(y)
a = 8
a = a + 1
print(a)

# !! Data types
# !Strings: sth_string=
#eg
a_string='1 2 3 4 5 6 7'
print(a_string)
print(type(a_string)) # meaning of _string
print(a_string[0])  # 0: the 1st word of a 'a_string
print(a_string[4])  # 4: the 4th word of a 'a_string
print(a_string[-1]) #-1: last 1st word of a 'a_string
print(a_string[-3])  #-3: last 3rd word of a 'a_string

# Exercise 5
my_string= 'Some text characters of my choice.'
m=7
# Find out how many times we can print the mth character
# before exceeding the string length
N=len(my_string) #len(): the string length(-chang du) : characters+space+fu hao
print(N)
print(N // m)
print(my_string[m - 1]) # the 6th character: e
print(my_string[2 * m - 1]) #the 2*7-1=13th : r - bao kuo kong ge
print(my_string[3 * m - 1])  #the 3*7-1=20th : space
print(my_string[4 * m - 1])

# !! Booleans:
    #only take one of two values: True or False

#eg:
a=True
b= False
print(type(b))  # type of True/False is bool
print(a and b)  # True and False: False -and
print(a or b)   # Trus or False : True -or
print(not a)   # not True : false
print((not a) or b)   # False or False: False

#eg:
x = 3
y = 4
print(x < y)    # 3<4:false
print(x == y)  # 3=4: false
print(x == y or 5 >= 2)    # (3=4: false) or (5 >= 2:true) : true
print(x != y)   # !:no; !=:not equal
z = x > y  # z = 3>4: false
print(z)

#  Exercise 6:
u=3
v=5
same_sign=((u>0 and v>0) or (u<0 and v<0) ) # or: same_sign=u*v>0
print(same_sign)

# !! Floating point numbers:
    #represent real numbers on a computer.
# eg
print(3.456 + 11.888)
print(99.9 / 0.1)
print(2.0 * 11.4)
print(1.5e-5 + 1.0e-6) # e-n: 10 ^(-n)
print(type(2.0 * 11.4)) # float
import numpy as np
print(np.sqrt(2.0))

# excerise 7
x = 1.11e-16
print(x == 0)
print(x + 1 == 1)
print(np.finfo(1.0))
np.finfo(1.0).eps
