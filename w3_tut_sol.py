## Conditional statements, more loops, and code review

# !! Conditional statements
# ! if statements
if my_condition:
    [some instructions]
#eg:
# Define some variables
a = 4.3
b = 5.1
c = 'hello'
i = 1
j = 8
z = True
# for compare i and j
if i == j:
    # This is not true -- any instructions
    # in this block are ignored
    print('i and j are equal')  # none result: condition false
if i < j:
    print('i is less than j')  # i is less than j: true condition
# consider i'type is an integer or bot
if type(i) == int:
    print('i is an integer') # i is an integer  : : true condition
# consider c'type is string and j'type not a float
if type(c) == str and type(j) != float:
    print('c is a string and j is not a float')  # c is a string and j is not a float :true condition
# condition: a+b > 7 has result; not without result
if (a + b) > 7:
    print(a + b)  #9.399999999999999
# if true have result
if z:
    print(a)  # 4.3
# Recall boolean casting in W1...
if j:
    print('j is not zero nor empty')  #j is not zero nor empty


# !if-elif-else blocks
if cond_1:
    # [some instructions, executed if cond_1 is true]
elif cond_2:
    # [other instructions, executed if cond_1 is false,
    # but cond_2 is true]
else:
    # [other instructions, executed if both cond_1 and cond_2
    # are false]
# eg
a = 4.9
b = 5.4
if a > b:
    print('a is greater than b')
elif a < b:
    print('a is smaller than b') #result a is smaller than b: satisfy this condition
else:
    print('a is equal to b')

 # Exercise 1
import numpy as np
rng = np.random.default_rng()
n = rng.integers(1, 1001)
# if a multiple of both 3 and 7
if n % 21 == 0:  # == : equal to
    print(n, 'is a multiple of both 3 and 7.')
# not a multiple of both, but a multiple of either
elif n % 3 == 0 or n % 7 == 0:
    print(n, 'is a multiple of one of 3 or 7.')
# the last possible case: not a multiple of either
else:
    print(n, 'is not a multiple of 3 nor 7.')


# Exercise 2:
zen = 'If the implementation is hard to explain, it is a bad idea. If the implementation is easy to explain, it may be a good idea.'
count = 0 #each words
# The .split() method returns a list of words
for word in zen.split():
#if it contains an e, print the word.
    if 'e' in word:
        print (word)
# if it does not contain an e, but contains an i, print the first character of the word.
    elif 'i' in word:
        print (word[0])
# if it does not contain an e nor an i, increment count by 1.
    else:
        count += 1
#Result
for：
 if:
    we
    the
    implementation
    i
    explain,
    i
    idea.
    the
    implementation
    i
    easy
    explain,
    i
    be
    idea.
return


# ! while loops
#   - while loops are used to repeat a set of instructions while a given condition is true. The while statement does not use any placeholder variables; instead, it must be given a Boolean object (i.e., an expression which evaluates to either True or False).
#   if condition is True
while my_condition:
 [some instructions]


#eg
S = 0
i = 0
# if i is less and equal to 10
while i <= 10:
    S += i
    i += 1
print(S) # 55 : 1+2+3+...+10

# Exercise 3
import math
n=0
e=1
while not math.isclose(math.exp(1), e, rel_tol=1e-6):
    n += 1                      # increment n by 1
    e += 1 / math.factorial(n)  # add the nth term of the series

    print(f'n = {n}')
    print(f'Exact value of exp(1): {math.exp(1):.6f}')
    print(f'Approximate value: {e:.6f}\n')

print(f'{n} iterations are needed.')
# result
for：
 if:
     55
     n = 1
     Exact value of exp(1): 2.718282
     Approximate value: 2.000000

     n = 2
     Exact value of exp(1): 2.718282
     Approximate value: 2.500000

     n = 3
     Exact value of exp(1): 2.718282
     Approximate value: 2.666667

     n = 4
     Exact value of exp(1): 2.718282
     Approximate value: 2.708333

     n = 5
     Exact value of exp(1): 2.718282
     Approximate value: 2.716667

     n = 6
     Exact value of exp(1): 2.718282
     Approximate value: 2.718056

     n = 7
     Exact value of exp(1): 2.718282
     Approximate value: 2.718254

     n = 8
     Exact value of exp(1): 2.718282
     Approximate value: 2.718279

     n = 9
     Exact value of exp(1): 2.718282
     Approximate value: 2.718282

     9 iterations are needed.

     55
     n = 1
     Exact value of exp(1): 2.718282
     Approximate value: 2.000000

     n = 2
     Exact value of exp(1): 2.718282
     Approximate value: 2.500000

     n = 3
     Exact value of exp(1): 2.718282
     Approximate value: 2.666667

     n = 4
     Exact value of exp(1): 2.718282
     Approximate value: 2.708333

     n = 5
     Exact value of exp(1): 2.718282
     Approximate value: 2.716667

     n = 6
     Exact value of exp(1): 2.718282
     Approximate value: 2.718056

     n = 7
     Exact value of exp(1): 2.718282
     Approximate value: 2.718254

     n = 8
     Exact value of exp(1): 2.718282
     Approximate value: 2.718279

     n = 9
     Exact value of exp(1): 2.718282
     Approximate value: 2.718282

     9 iterations are needed.


# ! break (The break statement in loops)
#   - Sometimes, we may wish to exit a loop early -- for example, when we try to find the first element in a list which matches a condition. Once we find the element, we don't want to keep looping through the rest of the list.
# eg
list_of_strings = ['hello', 'this', 'is', 'a', 'lot', 'of', 'text', 'in', 'a', 'list.']

# Find and display the first word which starts with an i
for word in list_of_strings:
    if word[0] == 'i':
        print(word) # is
# This stops the loop immediately
        break    # break: when find the first result, we will stop right now, stop and not continue

# Exercise 4
count = 0
for i in range(10): # The outer loop runs for 10 iterations
    for j in range(5): # at each of these, the inner loop runs for 5 iterations
        count += 1
        if count > 17:
            break
    print(count)
#Result
for:
    if:
        5
        10
        15
        18
        19
        20
        21
        22
        23
        24



# ! Debugging and troubleshooting
# Errors and exception
my_string = 'Hello world'
if my_string[0] == 'H'
    print(my_string) # SyntaxError(type of error): invalid syntax(error message)

# Built-in exception types
1. IndexError: a sequence subscript is out of range.
my_list = [1, 2, 3, 4]
print(my_list[4])   # IndexError: list index out of range:  we're trying to access my_list[4], but my_list only has elements up to my_list[3].

2. NameError: the variable referred to does not exist -- there is no box in memory with this label.
my_list = [1, 2, 3, 4]
print(my_ist) # NameError: name 'my_ist' is not defined, This often comes up when you mistype a variable name.

3. SyntaxError: the code is not syntactically correct --- it is not valid Python, so Python doesn't know how to interpret it.
my_string = 'Hello world'
if my_string[0] == 'H'
    print(my_string)

4. TypeError: a very common error to see when learning Python! This means that an operation or a function is applied to an object of the wrong type
# Trying to index something that is not a sequence...
my_int = 4
print(my_int[2])
# Trying to multiply two lists together...
my_list = [1, 2, 3, 4]
my_other_list = [5, 6, 7]
print(my_list * my_other_list)
# Trying to multiply two lists together...
my_list = [1, 2, 3, 4]
my_other_list = [5, 6, 7]
print(my_list * my_other_list)

5.
ValueError: raised when an operation or function is applied to an object with the right type, but an invalid value.
#For example, the int() function can cast a string to an integer, if the string can be interpreted as a number.
a = int('432')    # all good
b = int('hello')  # ValueError

# ! Testing
#   -When you write or review code, it's important to test it often and comprehensively.

# eg
def find_divisors(nums, n):
    '''
    Returns a list of all divisors of n
    present in the list nums.
    '''
    divisors = []
    for i in nums:

        print(f'Current number being tested is {i}.')
        print(f'Is {i} a divisor of {n}?')

        # Check if n/i is an integer
        print(f'{n} / {i} = {n / i}')

        if isinstance(n / i, int):
            print(f'Yes, adding {n} to the list\n')
            divisors.append(n)
        else:
            print('No\n')

    return divisors

# Test example: result should be [1, 1, 1, 1] (no matter the choice of n)
divisors = find_divisors([1, 1, 1, 1], 97)
print(f'Result: {divisors}\n')
# result:
for:
    if:
        `Current number being tested is 1.
        Is 1 a divisor of 97?
        Yes, adding 1 to the list (oops, not n!)

        Current number being tested is 1.
        Is 1 a divisor of 97?
        Yes, adding 1 to the list (oops, not n!)

        Current number being tested is 1.
        Is 1 a divisor of 97?
        Yes, adding 1 to the list (oops, not n!)

        Current number being tested is 1.
        Is 1 a divisor of 97?
        Yes, adding 1 to the list (oops, not n!)

        Result: [1, 1, 1, 1]

# Test example: result should be [1, 2, 3, 4, 6]
divisors = find_divisors([1, 2, 3, 4, 5, 6, 7, 8], 12)
print(f'Result: {divisors}\n')
# result:
for:
    if:
        result
        Current number being tested is 1.
        Is 1 a divisor of 12?
        12 / 1 = 12.0
        No

        Current number being tested is 2.
        Is 2 a divisor of 12?
        12 / 2 = 6.0
        No

        Current number being tested is 3.
        Is 3 a divisor of 12?
        12 / 3 = 4.0
        No

        Current number being tested is 4.
        Is 4 a divisor of 12?
        12 / 4 = 3.0
        No

        Current number being tested is 5.
        Is 5 a divisor of 12?
        12 / 5 = 2.4
        No

        Current number being tested is 6.
        Is 6 a divisor of 12?
        12 / 6 = 2.0
        No

        Current number being tested is 7.
        Is 7 a divisor of 12?
        12 / 7 = 1.7142857142857142
        No

        Current number being tested is 8.
        Is 8 a divisor of 12?
        12 / 8 = 1.5
        No

        Result: []


#Exercise 5
def find_divisors(nums, n):
    '''
    Returns a list of all divisors of n
    present in the list nums.
    '''
    divisors = []
    for i in nums:

        print(f'Current number being tested is {i}.')
        print(f'Is {i} a divisor of {n}?')

        # Check if n/i is an integer number, instead of an 'int' object
        if n % i == 0:
            print(f'Yes, adding {i} to the list (oops, not n!)\n')
            divisors.append(i)
        else:
            print('No\n')

    return divisors
# Test example: result should be [] for any number n smaller than all the list elements
divisors = find_divisors([10, 10, 10], 5)
print(f'Result: {divisors}\n')
#result:
for:
    if:
        `
        Current number being tested is 10.
        Is 10 a divisor of 5?
        No

        Current number being tested is 10.
        Is 10 a divisor of 5?
        No

        Current number being tested is 10.
        Is 10 a divisor of 5?
        No

        Result: []

# Test example: result should be [] for any prime number n and any list not containing 1 or n
divisors = find_divisors([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 23)
print(f'Result: {divisors}\n')
# result:
for:
    if:
    `Current number being tested is 2.
    Is 2 a divisor of 23?
    No

    Current number being tested is 3.
    Is 3 a divisor of 23?
    No

    Current number being tested is 4.
    Is 4 a divisor of 23?
    No

    Current number being tested is 5.
    Is 5 a divisor of 23?
    No

    Current number being tested is 6.
    Is 6 a divisor of 23?
    No

    Current number being tested is 7.
    Is 7 a divisor of 23?
    No

    Current number being tested is 8.
    Is 8 a divisor of 23?
    No

    Current number being tested is 9.
    Is 9 a divisor of 23?
    No

    Current number being tested is 10.
    Is 10 a divisor of 23?
    No

    Current number being tested is 11.
    Is 11 a divisor of 23?
    No

    Current number being tested is 12.
    Is 12 a divisor of 23?
    No

    Result: []

# Test example: result should be [1] for any prime number n and any list containing 1 but not n
divisors = find_divisors([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 17)
print(f'Result: {divisors}\n')
# Result
for:
    if:
        `Current number being tested is 1.
        Is 1 a divisor of 17?
        Yes, adding 1 to the list (oops, not n!)

        Current number being tested is 2.
        Is 2 a divisor of 17?
        No

        Current number being tested is 3.
        Is 3 a divisor of 17?
        No

        Current number being tested is 4.
        Is 4 a divisor of 17?
        No

        Current number being tested is 5.
        Is 5 a divisor of 17?
        No

        Current number being tested is 6.
        Is 6 a divisor of 17?
        No

        Current number being tested is 7.
        Is 7 a divisor of 17?
        No

        Current number being tested is 8.
        Is 8 a divisor of 17?
        No

        Current number being tested is 9.
        Is 9 a divisor of 17?
        No

        Current number being tested is 10.
        Is 10 a divisor of 17?
        No

        Current number being tested is 11.
        Is 11 a divisor of 17?
        No

        Current number being tested is 12.
        Is 12 a divisor of 17?
        No

        Result: [1]

# Test example: result should be [1, n] for any prime number n and any list containing 1 and n
divisors = find_divisors([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 11)
print(f'Result: {divisors}\n')
# Result
for:
    if:
        `Current number being tested is 1.
        Is 1 a divisor of 11?
        Yes, adding 1 to the list (oops, not n!)

        Current number being tested is 2.
        Is 2 a divisor of 11?
        No

        Current number being tested is 3.
        Is 3 a divisor of 11?
        No

        Current number being tested is 4.
        Is 4 a divisor of 11?
        No

        Current number being tested is 5.
        Is 5 a divisor of 11?
        No

        Current number being tested is 6.
        Is 6 a divisor of 11?
        No

        Current number being tested is 7.
        Is 7 a divisor of 11?
        No

        Current number being tested is 8.
        Is 8 a divisor of 11?
        No

        Current number being tested is 9.
        Is 9 a divisor of 11?
        No

        Current number being tested is 10.
        Is 10 a divisor of 11?
        No

        Current number being tested is 11.
        Is 11 a divisor of 11?
        Yes, adding 11 to the list (oops, not n!)

        Current number being tested is 12.
        Is 12 a divisor of 11?
        No

        Result: [1, 11]

# ! Code review: the essentials
# Code comments
#   -In Python, code comments start with a # character; anything on the same line after the # character is ignored by the Python interpreter.
#eg
# Create 2 integer variables a and b
a = 4
b = -8
# Find out whether a divides b
print(b % a == 0)
# Exercise 6
def fibonacci(n):
    '''
    Returns the Fibonacci sequence up to xn,
    starting with x0 = 1, x1 = 1, as a list.
    '''
    # Start a list x with the two initial values
    x = [1, 1]
    # The list will be [x0, x1, ..., xn], which is n+1 total elements.
    # Add the remaining n-1 elements with a loop
    for i in range(n-1):
        # The next element is the sum of the 2 previous elements
        x.append(x[i] + x[i+1])
    # Return x to
    return x
# Compute and display the Fibonacci sequence up to the 6th term
print(fibonacci(5))  # True [1, 1, 2, 3, 5, 8]

# !Code style
#   -Generally, the structure, variable name, and commenting choices are referred to as the style of your code. Style is important for code readability, and for consistency when working as part of a team.

# Whitespace
#eg

import numpy as np
#Some not very easily readable code...
a=(np.sqrt(5))
a +=2 *np.exp(4*np.pi)-np.sin (a**2)
s='I am a string!'
print(s[ 3 ])
# A little better...
a = np.sqrt(5)
a += 2 * np.cos(4 * np.pi**2) - np.sin(a**2)

# with a space in last line, which speparate two parts
s = 'I am a string!'
print(s[3])

# Exercise 7
import numpy as np
# Set polynomial coefficient values
a = 2
b = .5
c = -9
# Compute square root of discriminant
sqrt_delta = np.sqrt(b**2 - 4*a*c)
# Compute the roots
x1 = (-b - sqrt_delta) / (2*a)
x2 = (-b + sqrt_delta) / (2*a)
# Display the roots
print('First root: x1 =', x1)
print('Second root: x2 =', x2)
# Check the results
print('Is x1 correct?', a*x1**2 + b*x1 + c == 0)
print('Is x2 correct?', a*x2**2 + b*x2 + c == 0)
# results
for:
    if:
        `
        First root: x1 = -2.25
        Second root: x2 = 2.0
        Is x1 correct? True
        Is x2 correct? True


# ! Standard input:  input()
#       takes a string as an argument, which will be displayed as a prompt. The user will be prompted to type a value and press Enter -- this value will be returned as a str by input().

# eg
# Ask user to enter a number, assign it to a variable
your_number = input('Please enter a number and press Enter: ')
print(your_number) # a new window to inter sth
print(type(your_number)) #<class 'str'>
#results
for:
    if:
        `
        different input
        when input 1 shows
        1
        #<class 'str'>
        when input a
        a
        #<class 'str'>

# eg
your_number = float(input('Please enter a number and press Enter: ')) # float: 实行 in a {}
print(f'{your_number} divided by 3 is {your_number / 3}.') # 1.0 divided by 3 is 0.3333333333333333.

# Exercise 8
import numpy as np

rng = np.random.default_rng()
target = rng.integers(1, 101)

n=int(input('Please enter a number and press Enter: '))
# result is different each time
while True:  # True 一定要大写首字母
    if n == target:
        #{target} shows the result below what the target of n
        print (f'correct, n is {target}' ) #correct, n is (u enter number) {} shows the result of
        break  # stop in this process
    elif n > target:
        n=int(input('too big, again: '))
    else:
        n=int(input('too small, again: '))
