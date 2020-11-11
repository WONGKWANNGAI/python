### !!! Week 2 worksheet: Lists, loops, and functions

# !! Type-casting, duck-typing
# eg
print(type(39)) # int
print(type(5))  # int
print(39 / 5)       # 7.8
print(type(39 / 5)) # float

# ! Type-casting:
    #is the technical name for changing the type of an object.
#eg:
i = 2566
print(type(i)) # <class 'int'>
print(float(i))    # 2566.0
print(str(i))   # 2566
string_number = '273'
print(float(string_number))     # 273.0
print(int(string_number))   # 273
# eg
string_number = 'twenty'
print(int(string_number))  # error !!!

# Exercise 1:
# The bool() function casts an object to the Boolean type.
#   bool(): True/ False
i = 2566
print(bool(i)) # True

i = 0.0
print(bool(i)) # False

i = -3.4
print(bool(i)) # True

i = '273'
print(bool(i)) # True

i = '0'
print(bool(i)) # True

i = ''
print(bool(i)) # False

i = 'False'
print(bool(i)) # True


# !! Lists : [...,...,...]
    # A list can be defined by listing its elements inside square brackets [...], separated by commas.
#eg
a = [1, 2, 3, 10, 6]
b = ['Hi', 'how', 'are', 'you?']
c = ['my', 1, 4.5, ['you', 'they'], 432, -2.3, 33]
d = []
print(a)    # [1, 2, 3, 10, 6]
print(b[0])     #  b[0]: the 1st in b list; Hi
print(c[-3])    #  c[-3]): the last 3rd in c list; 423

# ! runing in list:
# Join two lists together
print(a + b)    # [1, 2, 3, 10, 6, 'Hi', 'how', 'are', 'you?']

# Find the length of a list
print(len(b))   # 4


# Append (add) an item to the end of a list
c.append('extra')   # .append(): append sth in list
print(c)  # ['my', 1, 4.5, ['you', 'they'], 432, -2.3, 33, 'extra', 'extra']


# Print the 2nd element of the 4th element of c
print(c[3][1])  # they : the 4th of at list c is ['you', 'they'], the 2nd one in the 4th in c


# Sort a list
print(sorted(a))   # [1, 2, 3, 6, 10]; sorted(): ! pai xu


# Create a list with 12 repetitions of the same sequence
print(12 * [3, 0, 'y']) # n*[]: create n times of []
    # #[3, 0, 'y', 3, 0, 'y', 3, 0, 'y', 3, 0, 'y', 3, 0, 'y', 3, 0, 'y', 3, 0, 'y', 3, 0, 'y', 3, 0, 'y', 3, 0, 'y', 3, 0, 'y', 3, 0, 'y']


# Check if something is in a list:
    #stn in list is true or false?
print('my' in c)    # True
print('you' in c)    # False
print('you' in c[3])    # True
print(7 in a)       # False

#Exercise 2:
# Create the mxn identity matrix : =[n*[1,...,m]]
my_mat=[[1,0], [0,1]]  # Create the 2x2 identity matrix

# fnc_sth[n].append(m): append one element m to the n-1  row
my_mat[0].append(0)     # Append one element 0 to the 1st rows
my_mat[1].append(0)     # Append one element 0 to the 2nd rows

# fnc_sth.append(m): append m at the last of fnc_sth
my_mat.append([0, 0, 1])  # Append the 3rd row to the list of rows
print(my_mat)   # [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# ! ! Slicing
# !Index slicing allows us to extract a new list from any subsequence of an existing list.

# eg
a = [2, 5, 4, 8, 8] # length a
#  l[start:stop]  from start to stop-1
print(a[1:3])  # [5, 4]: start at 2nd, end at 3rd (stop-1:3-1=2)

# l[start:]  from start to len(l)-1
print(a[2:])  #[4, 8, 8]: start at 3nd

# l[:stop] from 0 to stop-1
print(a[:-2])   # [2, 5, 4]: stop at the last 3rd

# l[start:stop:step] from start to stop-1, with increment step (number -1)
print (a[0:5:2]) #[2, 4, 8]: 1st to 5th, with 1 step

# l[::step] from 0 to len(l)-1, with increment step
print (a[::2])  #[2, 4, 8]: all elements, with 1 step

# l[::], l[:] all the elements
print (a[:])    #[2, 5, 4, 8, 8]
print (a[::])  # [2, 5, 4, 8, 8]

# Exercise 3:  Consider the list m below. What is the most concise way to create a new list m_back, which takes as value the list m, backwards? In other words, print(m_back) should display
m=['e', 'd', 'c', 'b', 'a']
m_back=m[:]
print (m_back)  # ['e', 'd', 'c', 'b', 'a']

# Exercise 4:
# We can get pi from the math module
import math
# Create a variable n, with with value n is N, 5<=n<=20
n=5
my_list= [1]*(n-1) # vextor with 1 variable [1] become n-1 variables
print (my_list) #[1, 1, 1, 1]
# append pi to the list
my_list.append(math.pi)
print (my_list) #[1, 1, 1, 1, 3.141592653589793]
print('Does the list have length n now?', len(my_list) == n) # Does the list have length n now? True

# Change the value of the 3rd element
print(len(my_list)) #5
my_list[2]=sum(my_list[3:]) #the sum of its last n-1 elements: sum of last 2 elements; the last 1st = the 4th element= len(m)-4+1=3
print(my_list) #[1, 1, 4.141592653589793, 1, 3.141592653589793]

# !! Loops
# -give the computer a set of instructions and a stopping point, and it will execute these instructions over and over until that point is reached.
# eg
my_string = 'a bc defg hijkl mnop'
m = 4
N = len(my_string)
print(N) # 20
# Loop over the characters
for i in range(m-1, N, m):
    print(my_string[i])
# result:
c #the 4th character
f #the 8th character
i #the 12th character
  #the 16th character
p #the 20th character


# ! for loops
#   -for loops iterate over the elements of a sequence (e.g. a list or a string), in the order in which they appear in that seque
for i in my_seq:
    [some instructions]
#eg for loops
a = [1, 2, 3, 10, 6]
for element in a:
    print(element)
print('These are all the elements of a.')
#result: shows in each steps
1
2
3
10
6
These are all the elements of a.

#Exercise 5: none

# !! Ranges
#   -There is another sequence type which we haven't mentioned so far, but is often useful in conjunction with for loops: the range type. A range is a sequence of increasing or decreasing integers, and can be created using the range() function
#eg
# range(j)             # 0, 1, 2, ..., j-1
print(range(5)) # range(0, 5): 0,1,...,5

# range(i, j)          # i, i+1, i+2, ..., j-1
print(list(range(5))) # [0, 1, 2, 3, 4]: 0, 1,..., 5-1=4

# range(i, j, k)       # i, i+k, i+2k, ..., i+m: from i to j, each element plus k
print(list(range(1, 10, 2))) # [1, 3, 5, 7, 9]: from 1 to 10, each elements increase 2

# sum with range
S = 0
# Loop over indices 0 to 5, inclusive
for i in range(6):  # 10+1=11
    S += i    # this is a shortcut for S = S + i: +=:sum
print(S) # 15 = 1+2+3+4+5

# loops with sequence
a = [2, 5, 7, 2, 1]
# Looping over the list by element
for element in a:
    print(element)
# Looping over the list by index
for idx in range(len(a)):
    print(a[idx])
# both result:
2
5
7
2
1

#  Exercise 6:
# Set n and initialise P
n=10
p=1
# Loop from j=2 to j=n j=n: n+1
for j in range(2,n+1):   #j=n: n+1
    p *= j**3 + 5*j**2 - 3 # *=: Multiply each term in succession
print (p) #19386192354630917063625 : (2^3+5*2^2-3)*(3^3+5*3^2-3)*...*(10^3+5*10^2-3)


# ! Defining functions
#   - You can also define your own custom functions to encapsulate specific subtasks and structure your programs. A function is essentially a block of code which only executes when the function is called.
def my_func(inputs): # my_func: the name of your function;  (inputs):the (zero or more) input arguments
    [function body] #the commands to execute upon calling the function
    return outputs  #  the (zero or more) return values or output values
# eg
import math
def f(x): # function named f, cariable is x
    y=(3*x-2)/math.sqrt(2*x+1) #A function object is created as a set of instructions
    return y # let y into f(x)
# Three number objects are created and respectively named a, b, c
a = 1
b = 2
c = math.pi
# Display return values on the screen
print(f(a), f(b), f(c))  #x=1: 0.5773502691896258; x=2: 1.7888543819998317 ; x=pi: 2.751203976799908

# Assign return values to variables
f_pi = f(c) #The function f is called with input value c; the instructions are executed. f returns a new number object, which we store in memory with the name f_pi.
print('f(x) evaluated at x = pi is', f_pi) #f(x) evaluated at x = pi is 2.751203976799908

# Returning multiple values
def sum_diff_prod(a, b):
    #To return multiple values from a function, we can list them after the return statement, one after another, separated by commas.
    '''
    Computes and returns the sum, difference,
    and product of a and b.
    '''
    return a+b, a-b, a*b
# Print the output of the function as a tuple
print(sum_diff_prod(12, 4)) #(16, 8, 48)
print(sum_diff_prod(0, -1.2)) #(-1.2, 1.2, -0.0)
# Unpack the output into different variables
s, r, m = sum_diff_prod(-4, -3) #s:sum a+b; r:reduce a-b; m:multiple a*b
print (s) # -7
print (r) # -1
print (m) # 12

# Exercise 7
# way 1:
def compute_P(n):
    p=1  # Initialise P !!!
 # Loop from j=2 to j=n
    for j in range(2,n+1):
        p *= j**3 + 5*j**2 - 3  # Multiply each term in succession
    return  p  #  # Return the result
print(compute_P(7)) # 13811904975375

# way 2:
def compute_P(n):
    '''
    Computes the product P for a value of n.
    '''
    # Initialise P
    P = 1

    # Loop from j=2 to j=n
    for j in range(2, n+1):
        # Multiply each term in succession
        P *= j**3 + 5*j**2 - 3

    # Return the result
    return P

# Test the function
print(compute_P(7))
