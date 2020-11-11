## !!! Week 4 worksheet: Namespaces, NumPy, and plotting with matplotlib

# !! Namespaces and scope
#  ! Namespace
#   -  a set of names, together with the objects they refer to.
# eg
a = 5
b = 2
print(a + b) # 7
print(d)  # NameError: name 'd' is not defined

# ! Scope
'''
#   - a part of your code from where you can access a certain namespace
 -- that is, where you can find the variables in that namespace by just typing their names.
'''
# eg
outside = 7 # global scope

def func():
    outside = 12
    # The name "outside" refers to 12 here locally...
    print(outside) #12

func()

# ...but here, "outside" still refers to the one defined globally
print(outside) #7

# !! Importing modules
'''
#       -There are different ways you can import a module
 -- you can also only import selected functions or attributes from a particular module,
  if you need just a few specific things, instead of importing the whole module.
'''
# eg
# Import a module as a separate namespace
import numpy
import numpy as np
print(np.cos(np.pi)) # -1.0


# Import something from a module into the main namespace
from numpy import cos, pi
print(cos(pi)) # -1.0

# !! Numpy arrays: 排列 （矩阵）
'''
a list as an input argument, and returns an array where the rows are the elements of the list.
'''
# eg
# Start by importing Numpy
import numpy as np
# Create a vector
v = np.array([3, 4, -2.8, 0]) # np.array():排列
print(v)
print(type(v))
# Create a matrix: pass a list of lists to np.array(),
# each element of which is a row of the matrix
id_4 = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]) # np.array():排列 （矩阵)
print(id_4)

# Use the second (optional) input argument of np.array()
# to specify the type of its elements:
id_4_float = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=float) #dtype=float: type of array is float
print(id_4_float)

# ! Some useful functions to construct arrays
# eg
# Create a matrix of zeros
A = np.zeros([3, 7])  # matrix 3*7 (row*column), all element is 0
print(A)
while:
        `result
        [[0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0.]]
# Create a vector of ones
u = np.ones(5) # matrix 1*5 (row*column), all element is 1
print(u)    # [1. 1. 1. 1. 1.]
# Create the 4x4 identity matrix, as above
id_4_mat = np.eye(4) # matrix 1*4 (row*column) is I matrix
print(id_4_mat)
while:
        `result
        [[1. 0. 0. 0.]
        [0. 1. 0. 0.]
        [0. 0. 1. 0.]
        [0. 0. 0. 1.]]
# Create a matrix of pseudo-random numbers between 0 and 1,
# sampled from a uniform distribution
B = np.random.random([3, 3])    # .random: number is different everytime
print(B)    # matrix 3*3
# Create a 1D array with a range of floats
v = np.arange(3.1, 5.2, 0.3)    # Start from 3.1 to 5.2, step of imcrease of 0.3
print(v)    # [3.1 3.4 3.7 4.  4.3 4.6 4.9 5.2]

# ! .shape:
'''
# Retrieve the dimensions of an array, as a tuple
'''
print(A.shape)  # (3, 7); A.shape: check the array of A


# First way: giving the list of rows explicitly
M1 = np.array([[0.1, 0.2, 0.3],
               [0.4, 0.5, 0.6],
               [0.7, 0.8, 0.9]])
print(M1)
# ! .reshape(n,m) : set a matrix of n*m
# Second way: using range() and .reshape()
# Note that range() returns a sequence, which we can therefore use
# directly as the input argument for np.array()
M2 = 0.1 * np.array(range(1, 10)).reshape((3, 3))
print(M2)
while:
    `result
                [[1. 0. 0. 0.]
                [0. 1. 0. 0.]
                [0. 0. 1. 0.]
                [0. 0. 0. 1.]]

# Third way: using np.arange() and .reshape
M3 = np.arange(0.1, 1, 0.1).reshape((3, 3))
print(M3)
while:
        `result
        [[0.1 0.2 0.3]
        [0.4 0.5 0.6]
        [0.7 0.8 0.9]]

#Exercise 1
import numpy as np
M= np.array([[9,3,0],[-2,-2,1],[0,-1,1]])
# result
while:
    `result
    [[ 9  3  0]
     [-2 -2  1]
     [ 0 -1  1]]
y= np.array([0.4, -3, -0.3])
print(M)
print(y) # [ 0.4 -3.  -0.3]

# ! Element-wise operations
'''
- operators +, -, *, /, and ** can be used to perform element-wise operations between:
 two arrays of the same size; or, an array and a scalar.
'''
# eg
# Yet another way to construct the matrix from earlier...
A = 0.1 * np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
print("A = ")
print(A)
while:
    `result
    A =
    [[0.1 0.2 0.3]
     [0.4 0.5 0.6]
     [0.7 0.8 0.9]]
B = np.eye(3)
print(B)
while:
    `result
    [[1. 0. 0.]
    [0. 1. 0.]
    [0. 0. 1.]]

print("\nA + B =")
print(A + B)
while:
    `result
        A + B =
        [[1.1 0.2 0.3]
         [0.4 1.5 0.6]
         [0.7 0.8 1.9]]

print("\nA - B =")
print(A - B)
while:
    `result
        A - B =
        [[-0.9  0.2  0.3]
         [ 0.4 -0.5  0.6]
         [ 0.7  0.8 -0.1]]

print("\nA * B =")
print(A * B)
while:
    `result
        A * B =
        [[0.1 0.  0. ]
         [0.  0.5 0. ]
         [0.  0.  0.9]]

print("\nA ** 2 =")
print(A ** 2)
while:
    `result
        A ** 2 =
        [[0.01 0.04 0.09]
         [0.16 0.25 0.36]
         [0.49 0.64 0.81]]

print("\n B / A =")
print(B / A)
while:
    `result
         B / A =
        [[10.          0.          0.        ]
         [ 0.          2.          0.        ]
         [ 0.          0.          1.11111111]]

# Exercise 2
import numpy as np
def dot_prod():
    '''
    Returns the dot product of vectors u and v.
    '''
    return np.sum(u * v) # the sum of u*n

def dot_prod(): # my solution, not sure is true?
    for i in range(n-1):
        uv +- u*v
        return uv

# ! Matrix operations and linear algebra
np.matmul() # its operator alias, @) allows to compute matrix products
# eg
A = 2 * np.ones([2, 4])
print("A =")
print(A)
while:
    `result
    A =
    [[2. 2. 2. 2.]
     [2. 2. 2. 2.]]`

B = 0.4 * np.eye(4)
print("B =")
print(B)
while:
    `result
    B =
    [[0.4 0.  0.  0. ]
     [0.  0.4 0.  0. ]
     [0.  0.  0.4 0. ]
     [0.  0.  0.  0.4]]
v = np.random.random(4)
print("v =")
print(v)  # matrix 1*4 with different results everytime
# Products AB and BA^T
# .matmul(M,N) : matrix M*N
print(np.matmul(A, B)) # A*B
while:
        `result`
        [[0.8 0.8 0.8 0.8]
         [0.8 0.8 0.8 0.8]]
print(np.matmul(B, A.T)) # B* A; T notation used to transpose arrays.
while:
    `result`
    [[0.8 0.8]
     [0.8 0.8]
     [0.8 0.8]
     [0.8 0.8]]
print(np.matmul(B, A.transpose())) # .transpose()= .T 矩阵转置
while:
    `result`
    [[0.8 0.8]
    [0.8 0.8]
    [0.8 0.8]
    [0.8 0.8]]

# Products Av and v^T B
print(np.matmul(A, v))  # [6.2277018 6.2277018]: matrix A*v
print(np.matmul(v, B))   # [0.31996841 0.18086158 0.39364848 0.35106189] : : matrix v*B

# Dot product of v with itself
print(np.matmul(v, v))  # 2.583089203617617 :matrix v*v

# We can also use the operator @ to do exactly the same thing:
#   M @ N = np.matmul(M,N) : matrix M*N
print(B @ A.T) # B* A
while:
    `result`
    [[0.8 0.8]
    [0.8 0.8]
    [0.8 0.8]
    [0.8 0.8]]
print(A @ v) # [6.2277018 6.2277018]: matrix A*v
print(v @ B)    # [0.31996841 0.18086158 0.39364848 0.35106189] : : matrix v*B
print(v @ v)    # 2.583089203617617 :matrix v*v

# linalg
'''
 -Numpy has a sub-module called linalg, which contains many useful functions for linear algebra and matrix operations.
 linalg linalg=linear+algebra，norm则表示范数，首先需要注意的是范数是对向量（或者矩阵）的度量，是一个标量（scalar）
np.linalg.inv()：矩阵求逆; np.linalg.det()：矩阵求行列式（标量）
'''
# eg
# Create a random 3x3 matrix and a vector of three 1s
A = np.random.random([3, 3])
b = np.ones(3)
print (A)
while:
    `result`
    [[0.99952246 0.5463301  0.00336856]
     [0.89330684 0.33236699 0.23357147]
     [0.1510094  0.06681395 0.07677958]]
# Eigenvalues of a matrix: note the complex values here, j=sqrt(-1) 调用eigvals函数求解特征值
print(np.linalg.eigvals(A))  # [ 1.45272287 -0.10058847  0.05653464]
# Eigenvalues and right eigenvectors
'''
使用eig函数求解特征值和特征向量。该函数将返回一个元组，按列排放着特征值和对应的特征向量，其中第一列为特征值，第二列为特征向量。
'''
eig_val_A, eig_vec_A = np.linalg.eig(A)
print("Eigenvalues: ", eig_val_A)   # Eigenvalues:  [ 1.45272287 -0.10058847  0.05653464]
print("Eigenvectors: ", eig_vec_A)
while:
        `result
        Eigenvectors:  [[-0.7649222  -0.44450638 -0.37323351]
                         [-0.63382301  0.89481974  0.64007537]
                         [-0.11472759  0.04137275  0.67156553]]
print('\nQR and SVD:')
Q, R = np.linalg.qr(A)       # Q-R matrix decomposition Q-R矩阵分解：
print("Q =", Q)
while:
    `value`
    Q = [[-0.74092667  0.66885431 -0.06051104]
         [-0.66219108 -0.7426085  -0.10017778]
         [-0.11194035 -0.03415451  0.9931278 ]]
print("R =", R)
while:
    `value`
    R = [[-1.34901671 -0.63236018 -0.16575954]
         [ 0.          0.11631469 -0.17382145]
         [ 0.          0.          0.05264943]]
U, S, V = np.linalg.svd(A)   # Singular value decomposition 奇异值分解：使用svd函数分解矩阵
print("U =", U)
while:
    `value`
    U = [[-0.75410039  0.65195183  0.07931847]
         [-0.6466122  -0.71586717 -0.26348976]
         [-0.11500115 -0.24998602  0.96139572]]
print("S =", S) # S = [1.49938062 0.21189217 0.02600272]
print("V =", V)
while:
    `value`
    V = [[-0.89952452 -0.42323074 -0.10831148]
         [-0.12081043  0.47924251 -0.86932816]
         [-0.41983387  0.76889684  0.48222108]]

# Exercise 3.
import numpy as np
M= np.array([[9,3,0],[-2,-2,1],[0,-1,1]])
y= np.array([0.4, -3, -0.3])
x=np.linalg.solve(M,y)  # matrix y*M^(-1) ; .linalg.solve() 调用solve函数求解线性方程组
print(x) # [-2.56666667  7.83333333  7.53333333]

# ! Indexing and slicing Numpy arrays (索引)
#   - access an element in a Numpy array in a given position, we can use indexing, just like for other sequences.
v[i]  # the i+1th element of the vector v.
A[i, j]  # element in the i+1 row and j+1 column of the matrix A.
X[i, j, k, h, ...] # used to index elements for tensors in higher dimensions. 用于索引高维张量的元素
# eg
# Create a 4x4 matrix : ().reshape()
A = np.arange(1/4, 17/4, 1/4).reshape((4, 4))
print(A)
while:
    `result`
    [[0.25 0.5  0.75 1.  ]
     [1.25 1.5  1.75 2.  ]
     [2.25 2.5  2.75 3.  ]
     [3.25 3.5  3.75 4.  ]]
# Print some slices
print(A[1, 3]) # 2.0
 # A[0, :]: all elements at row 0
print(A[0, :])    # [0.25 0.5  0.75 1.  ]
# A[:, 2]: all elements at columns 2
print(A[:, 2])    # [0.75 1.75 2.75 3.75]
# A[2: , :-1]: rows 2 to the last, columns 0 to the second-to-last
print(A[2: , :-1])
while:
    `ressult`
    [[2.25 2.5  2.75]
    [3.25 3.5  3.75]]
# A[0::2 , 1]:every second row starting from 0 and step at 1, column 1
print(A[0::2 , 1]) # [0.5 2.5]

# Exercise 4
# Create a random number generator
rng = np.random.default_rng()
# Create a random 3x5 matrix of integers between 1 and 10
A = rng.integers(1, 11, size=[3, 5]) # matrox 3 rows and 5 columns
print(A, '\n')
# Create a Boolean array the same shape as A, with True where the corresponding
A5 = A < 5 # check elements in A is less than 5, true or false
print(A5, '\n')
# Use A5 to return all elements of A smaller than 5 (in a 1D array)
print(A[A5], '\n')
# Display the rows of A starting at row 1, and columns ending at column 2
print(A[1:, :3], '\n')
# Display the elements of that sub-matrix which are smaller than 5
print(A[1:, :3][A5[1:, :3]], '\n')
# Reassign all elements of A which are greater than or equal to 5 with the value 100
A[np.logical_not(A5)] = 100
print(A)

# Exercise 5
import numpy as np
n=2000
# Create a random number generator
rng = np.random.default_rng()
# Create a random matrix A with 2000x2000 elements between -1 and 1.05
n=2000
A=(1+1.05)*rng.random([n, n]) - 1.
# Get the sum of all rows of A
row_sums = np.sum(A, axis=1)
# Display the proportion of rows with a positive sum
positive_sum_rows = np.sum(row_sums >= 0)
print(f'The probability that a row of A is positive',
      f'is approximately {100 * positive_sum_rows / n : .1f}%.') # different result everytime, all over 95%

# Exercise 6
import numpy as np
n=4
# Initialise A with zeros
A = np.zeros([n,n])
# Loop over the rows...: i and j is in range from 0 to n-1
for i in range(n):
    for j in range(n):
        if i < j:
            A[i, j]= i + 2*j
        else:
            A[i, j]= i*j
print (A)
while:
    `result
    [[0. 2. 4. 6.]
     [0. 1. 5. 7.]
     [0. 2. 4. 8.]
     [0. 3. 6. 9.]]

# !! Plotting with matplotlib.pyplot
# ! A first plot
%matplotlib notebook
import matplotlib.pyplot as plt

# eg
import numpy as np
#   % matplotlib notebook
import matplotlib.pyplot as plt
# Create an x-axis with 1000 points, from 0 to 2 pi
x = np.linspace(0., 2*np.pi, 1000)
# Evaluate the function at all these points
y = x * np.sin(x) # y = x*sin(x)
# Create the plot and display it
plt.plot(x, y, 'k-')
plt.show() # result: a graph shows

# Exercise 7
import numpy as np
#   % matplotlib notebook
import matplotlib.pyplot as plt
# Create an x-axis with 1000 points, from -pi to pi
x = np.linspace(-1*np.pi, 1*np.pi, 1000)
# Evaluate the functions at all these points
f_1 = np.sin(x)
f_2 = np.tan((49/100)*x)
f_3 = np.sin(x)*np.cos(2*x)
# Create the plots in the same axes
plt.plot(x, f_1,  'r-.') # 'r-.' : red line
plt.plot(x, f_2,  'g:')   # 'g:' : green line
plt.plot(x, f_3,  'b--')  # 'b--': blue line
# Display the plot
plt.show()  # a graph with 3 colours lines

# ! Figures and axes as objects
 fig, ax = plt.subplots(m, n)
 '''
 creates a figure object, which we assign to the variable fig, and an array of axes,
 assigned to the variable ax, tiled in m rows and n columns
plt.subplots()是一个函数，返回一个包含figure和axes对象的元组
因此，使用fig,ax = plt.subplots()将元组分解为fig和ax两个变量。
下面两种表达方式具有同样的效果，可以看出fig.ax = plt.subplots()较为简洁。
'''
#eg
import numpy as np
import matplotlib.pyplot as plt

# Define x
x = np.linspace(0, 2*np.pi, 1000)

# Create figure and axes
fig_1, ax_1 = plt.subplots(2, 3)  # in fig_1: variabl1 is ax_1; totla figure: (2, 3) 2 rows * 3 columns

# We can also create a second figure, with the optional
# "figsize" argument of plt.subplots()
fig_2, ax_2 = plt.subplots(1, 4, figsize=(7, 2))

# Plot on 1st row, 2nd column of the first figure
ax_1[0, 1].plot(x, np.sin(x), 'm-') # [0, 1]: at 0 row, 2nd column. show sinx

# Plot on 2nd row, 3rd column
ax_1[1, 2].plot(x, np.cos(x), 'c-')

# Plot on 4th plot of the second figure
ax_2[3].plot(x, np.sinc(x), 'g-.')

# Update the display
plt.show()

#Exercise 8
'''Use the plt.subplots function to plot the three functions in the previous exercise in one figure,
with a different subplot for each.
'''
import numpy as np
import matplotlib.pyplot as plt
# Create an x-axis with 1000 points
x = np.linspace(-np.pi, np.pi, 1000)
# Evaluate the functions at all these points
f1=np.sin(x)
f2=np.cos(x)
f3=np.sinc(x)
# Create the plots in the same axes: f1,f2,f3 show in axes x
plt.plot(x,f1, 'r-.')
plt.plot(x,f2, 'g:')
plt.plot(x,f3, 'b--')
# Display the plot
plt.show()

# !! Basic reading and writing data from/to an external file
open() & .close()
'''
In the directory containing this notebook you should find the text files mytextfile.txt.
In Python, we can open files by creating a file object with a name.
It is also very important to close the file when we are done with it to avoid losing data
or corrupting our files.
'''
#eg
# open():First, we open our file with the Python function ,
# myfile = open(): assign it as a file object to the name myfile
myfile = open('mytextfile.txt','r')
# .read(): read its content, and save it in a variable contents.
contents = myfile.read()
# .close(): we close our file using the method
myfile.close()
# The content of this file can now be viewed by printing the variable contents.
print(contents)
'''
result:
Hello I am text for your Python Programming Course. This is my first line.
I want to tell you: YOU ARE WONDERFUL! Now I am finishing my second line.
It is rude not to say goodbye, so I am finishing my third line by saying BYE!
'''


# ! Using the with statement
# with...
'''
open() and .close(): Instead of using open() and .close() to open and close a file each time we want to access it,
with: we can use the with statement as shown below,
which automatically closes the file when the block ends.
.readline(): We can also read the file line by line by using the .readline() method of file objects:
'''
# eg
# with open('x.txt','r') as m: open file of x.txt as name of m
with open('mytextfile.txt','r') as myfile:
    # lines_content = m.readline() :  read its content, and save it in a variable contents
    lines_content = myfile.readline()
print(lines_content)
'''
result:
Hello I am text for your Python Programming Course. This is my first line.
'''

# Exercise 9
'''
Complete the following code to read all the lines in the file one by one until the end of the file,
and store them in the list all_lines.
You can either use .readlines() (search the documentation!),
or use a while loop to read the text line by line using .readline().
'''
# first approach: using 'readline'
with open('mytextfile.txt','r') as myfile:
    # Initialise an empty list to store the lines
    all_lines = []
    # if line is empty, the end of file is reached
    while True:
        # use readline to read the next line..
        line= myfile.readlines()
     # Break the loop when we reach an empty line (remember Boolean casting!)
        if not line:
            break
    # Append to the list if we still have non-empty lines
        all_lines.append(line)
        print(all_lines)

# second approach: using 'readlines'
with open('mytextfile.txt','r') as myfile:
    all_lines = myfile.readlines()
print(all_lines)
'''
result:
['Hello I am text for your Python Programming Course. This is my first line.\n',
'I want to tell you: YOU ARE WONDERFUL! Now I am finishing my second line.\n',
'It is rude not to say goodbye, so I am finishing my third line by saying BYE!']
'''

# ! Writing to a file: .write()
'''
In Python we can create a new file and open it, by using open() with the argument 'w' (meaning "write mode") to be able to write into it.
We also need to write the output into the file rather than display it on the screen.
Here is how we do it -- using the .write() method of file objects
'''
#eg
text1 = 'Here is the text to write into the file.'
text2 = 'Here is the second line of text to write into the file.'
# mynewfile.write(text1) : ombine text1 & text2 in file mynewfile in a line
with open('mytextfile_new.txt', 'w') as mynewfile:
    mynewfile.write(text1)
    mynewfile.write(text2)
'''
result:
a txt file name mytextfile_new is created
'''
text1 = 'Here is the text to write into the file.'
text2 = 'Here is the second line of text to write into the file.'
# mynewfile.write(text1 + '\n') : ombine text1 & text2 in file mynewfile in different
# text1 + '\n' : text1 at an individual line
with open('mytextfile_new1.txt', 'w') as mynewfile:
    mynewfile.write(text1 + '\n')
    mynewfile.write(text2 + '\n')

# Exercise 10:
'''
Write a script that reads the content of mytextfile.txt line by line,
and adds a line number followed by ': ' before each line.
 For example, the first line turns into
'1: Hello I am text for your Python Programming Course. This is my first line'
After adding the line numbers, the script should write all the new lines in a new file textfile_linenumber.txt.
Your final file should contain three lines.
'''
# Read the file, store the lines in a list
with open('mytextfile.txt', 'r') as myfile:
    all_lines=myfile.readlines()
# Edit and write each new line to a new file
with open ('textfile_linenumber.txt', 'w') as file:
    #  count():方法用于统计字符串里某个字符出现的次数
    count=1
    for line in all_lines:
        file.write(f'{count}:{line}')
        count += 1
'''
f'{count}:{line}':
https://blog.csdn.net/sunxb10/article/details/81036693
'''
