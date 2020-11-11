#   !!! Week 5 worksheet: Better plots, docstrings, and algorithm design


#  !!  Customising plots: 自定义图
# ! Setting plot axis properties 设置绘图轴属性
# Axis limits
.set_xlim() # set range of x
.set_ylim() # set range of y
'''
axis range on your plots can be controlled
'''
# eg
'''
plots the function sin(x)
my_ax :Axes object assigned to the variable
x-axis range: 0 ~ 2pi
y-axis range:  -1.1 ~ 1.1
'''
import matplotlib.pyplot as plt
import numpy as np
# Create 100 x-values from 0 to 2*pi
x = np.linspace(0, 2*np.pi, 100)
# Create the figure and axes
my_fig, my_ax = plt.subplots()
# Plot sin(x)
my_ax.plot(x, np.sin(x), 'k-')
# Adjust the x-axis and y-axis limits to tidy up the plot
# .set_xlim() : set range of x
my_ax.set_xlim([0, 2*np.pi])
# .set_ylim() : set range of y
my_ax.set_ylim([-1.1, 1.1])
# Show the figure
plt.show()

# ! Axis labels : Axis labels can be added
 .set_xlabel()
 .set_ylabel()
 # eg
ax.set_xlabel('x', fontsize=12)
'''
'x' :  a string.
 fontsize=12 : control the font size of the axis label, font size to 12pt.
'''
 # If you wish you can use LaTeX in axis labels via
ax.set_xlabel(r'x', fontsize=12)
'''
r : means "raw string";
 this allows backslashes in e.g. LaTeX math symbols to be interpreted correctly.
 '''

# Exercise 1:
'''
 Start by pasting your code for Exercise 8 in the Week 4 worksheet
 create the 3 plots again.
 Use set_xlim to control the x-axis range for each of the 3 plots,
 plotting values of x in [-pi, pi ]
 Use set_xlabel and set_ylabel to add axis labels to the plots,
 with a font size of 14pt
 '''
import matplotlib.pyplot as plt
import numpy as np
# Create an x-axis with 1000 points
x=np.linspace(-np.pi, np.pi, 1000)
# Evaluate the functions at all these points
f1 = np.sin(x)
f2 = np.tan(0.49 * x)
f3 = np.sin(x) * np.cos(2*x)
# Create a figure with 3 subplots
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
# Plot each function in a different subplot
ax[0].plot(x, f1, 'r-.')
ax[1].plot(x, f2, 'g:')
ax[2].plot(x, f3, 'b--')
# Store y-axis label for each plot
y_labels = [r'$f_1(x)$', r'$f_2(x)$', r'$f_3(x)$']
# Set all 3 properties for the 3 plots
for i in range (3):
    # .set_xlim([ , ]): set renge of x
    ax[i].set_xlim([-np.pi, np.pi])
    # .set_xlabel(r'$x$',fontsize= ):add y axis lables ; r : means "raw string"; fontsize= : a font size
    ax[i].set_xlabel(r'$x$', fontsize=14)
    # .set_xlabel( y_labels[i] ,fontsize= ):add x axis labels for y_labels[i] ; fontsize= : a font size
    ax[i].set_ylabel(y_labels[i], fontsize=14)
# Make some space
plt.subplots_adjust(hspace=0.5, wspace=0.5)
# Display the plot
plt.show()

# ! Adding a legend 添加图例
ax.legend(x,y,' ', label=r'$y$' )
'''
ax.legend(): dd a legend to the plot, using ax.legend(),labels all curves of the plot in ax.
The label text should be set when plotting the curve, using the label= keyword argument of .plot(),
and can contain LaTeX code.
'''
# eg
'''
a legend on ax with the green curve labelled t1 and the yellow curve labelled y2
'''
# Create an x-axis, and make 2 linear functions of x
x = np.linspace(-3, 3, 100)
y1 = 3*x - 2
y2 = -0.5*x + 1.5
# Plot both curves on the same axes
fig, ax = plt.subplots()
ax.plot(x, y1, 'g-', label=r'$y_1$')
ax.plot(x, y2, 'y-', label=r'$y_2$')
# .legend() will use the "label" arguments for each curve
ax.legend(loc='lower right', fontsize=14)
plt.show()

# Exercise 2
'''
The Maclaurin series for cos(x)
Create a new figure with a single set of axes.
cos(x) :  interval [-pi, pi]
plot the Maclaurin series truncated to the 2nd, 4th, and 6th order terms, evaluated over the same interval.
Set the axis ranges and labels, and add a legend to the plot.
'''
import matplotlib.pyplot as plt
import numpy as np
import math
# Define a function for the truncated Maclaurin series
def func_cos(x, n):
    '''
    Return the truncated Maclaurin series for
    cos(x), with terms up until order n.
    '''
    cos_series = 0
    for k in range(n//2 + 1):
        # Add each term of the series up to nth order
        cos_series += (-1)**k * x**(2*k) / math.factorial(2*k) # math.factorial(2*k) :  (2k)!

    return cos_series

# Create an x-axis with 1000 points; cos(x) :  interval [-pi, pi]
x = np.linspace(-np.pi, np.pi, 1000)
# Create a figure
fig, ax = plt.subplots()

# Plot the requested functions
ax.plot(x, np.cos(x), 'k-', label=r'$\cos(x)$')
ax.plot(x, func_cos(x, 2), 'r--', label=r'Order 2')
ax.plot(x, func_cos(x, 4), 'g-.', label=r'Order 4')
ax.plot(x, func_cos(x, 6), 'b:', label=r'Order 6')
# Set axis properties
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.5, 1.5])
ax.set_xlabel(r'$x$', fontsize=12)
ax.legend()
plt.show()

 Exercise 3:
 '''
 Reproduce the following figure, as closely as possible.
 (Note: the triangles are called "markers" -- you can search for that in the documentation.)
 '''
 import matplotlib.pyplot as plt
 import numpy as np
# Let's write a convenience function
def f(x):
    # Set coefficients
    a, b, c = -1, 3, 5
    # Compute the roots
    sqrt_delta = np.sqrt(b**2 - 4*a*c)
    root = [(-b - sqrt_delta)/(2 * a), (-b + sqrt_delta)/(2 * a)]
    # Return the polynomial and the 2 roots
    return a*x**2 + b*x + c, roots
# Create an x-axis, compute f(x) and both roots
x = np.linspace(-4, 5, 100)
y, root = f(x)
# Create the figure and axes
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
# Plot the function and the roots
ax.plot(x, y, '--', color='deepskyblue', label=r'$f(x) = -x^2 + 3x + 5$')
ax.plot(x, np.zeros(x.shape[0]), 'k-', label=r'$y = 0$')
ax.plot(root[0], 0, 'magenta', label='First root', marker='^', markersize=10)
ax.plot(root[1], 0, 'magenta', label='Second root', marker='^', markersize=10)
# Tidy up the plot
ax.set_xlim([-4, 5])
ax.set_ylim([y[0], 10])
ax.set_xticks(range(-4, 6))
ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$f(x)$', fontsize=14)
ax.set_title('Polynomial roots', fontsize=14)
ax.legend(loc='lower center')
ax.grid(True)

#   !! Docstrings 字串: '''
'''
a brief (1 or 2 sentences) description of what the function does,
a description of all input arguments and their type,
a description of all outputs and their type.
'''
# eg
def first_and_last(x):
    '''
    Returns the first and last characters of a string.

    Input:
        x (str): the input string.

    Output:
        out (str): a new string made of the first and
        the last characters of x.
    '''
    out = x[0] + x[-1]
    return out
print(first_and_last('Things'))
help(first_and_last)


#   !!  Designing algorithms设计算法
#   !   Recursion 递归
'''
a technical name for a procedure or function that calls itself in order to provide an answer.
为提供答案而调用自身的过程或函数的技术名称。
'''
# eg
def fac(n):
    '''
    Calculate n! for a positive integer n, using a recursive method.
    '''
    if n == 0:
        print('We got to the bottom...')
        return 1
    else:
        # Here, we call the function itself back with a different argument
        print(f'Now, n = {n}')
        return n * fac(n-1)

print(fac(3))
import math
print(math.factorial(3))

# Exercise 4:
'''
Write a recursive function fib_rec() which takes 3 input arguments:
a positive integer p,
a positive integer q,
a positive integer n greater than 2,
and returns the nth element of the (p,q)-Fibonacci sequence F(n) from the Week 2 workshop
F(1) = F(2) = 1,
F(n) = pF(n-1) + qF(n-2), for n>2
'''
def fib_rec(p, q, n):
    '''
    input : a positive integer p,
            a positive integer q,
            a positive integer n greater than 2
    output: if n=1 & 2,  fib_rec(n,p,q) = 1
            otherwise,fib_rec(n,p,q)= pF(n-1) + qF(n-2);
    '''
    if n == 1 or n == 2:
        return 1
    else:
        return p*fib_rec(p, q, n-1)+q*fib_rec(p, q, n-2)
print(fib_rec(1, 1, 15))    # 610
print(fib_rec(6, 4, 10))    # 5330944
print(fib_rec(2, 1, 35))    # 5168247530883

#   !!  A process for problem-solving
#   eg
'''
Problem-solving : the Post Office problem
Consider a 5*5km square city, with n post offices scattered at different (known) locations,
and all roads are laid out in a grid.
 Produce a map to help the residents find the post office within the shortest walking distance from any point in the city.
 You are given the following function to create the n post offices at random locations.
 '''
import numpy as np
def create_POs(n):
    '''
    Create random coordinates for n post offices in the city.

    Input: n (int), the total number of POs.
    Output: POs (ndarray), random array of size nx2, each row giving
        the (x, y) coordinates for one post office.
    '''
    # Initialise our random number generator :初始化我们的随机数生成器
    rng = np.random.default_rng()
    # Produce a random array of floats between 0.2 and 4.8 (not too close to city borders)
    POs = 0.2 + 4.6*rng.random(size=[n, 2])
    return POs
import matplotlib.pyplot as plt
# Create 10 post offices
n = 10
POs = create_POs(n)
# Plot their location
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(POs[:, 0], POs[:, 1], 'k*', markersize=15)
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
plt.show()

#   !!  Designing the algorithm 设计算法
np.random():  generate random (x, y) coordinates for all the residents
np.linalg.norm() : calculate Manhattan distances between points
# eg
def make_residents(pop):
    '''
    Creates a uniformly distributed population of city
    residents on the map.
    Input: pop (int): positive integer, number of residents (population).
    Output: residents (ndarray): Numpy array with pop rows
            and 2 columns, each row is the (x, y) coordinates
            of 1 resident.
    '''
    # Initialise our random number generator :初始化我们的随机数生成器
    rng = np.random.default_rng()
    # Create random (x, y) coordinates for "pop" residents, between 0 and 5
    residents = 5 * rng.random(size=[pop, 2])
    return residents
# A quick test to check that everything looks good for now
print(make_residents(5))
'''
result:
[[4.54167639 4.48251705]
 [3.27564085 3.99666343]
 [0.54355005 4.97249246]
 [2.92642008 1.2411141 ]
 [0.81554111 3.43062477]]
'''

np.linalg.norm(): 绝对值计算 | |
np.linalg.norm(res - po, ord=1)： |X-Xpo| + |Y-Ypo|

# eg
def nearest_PO(residents, POs):
    '''
    Finds the closest post office to all residents.
    Input:
        residents (ndarray): array with "pop" rows and 2 columns,
            each row is the (x, y) coordinates of 1 resident.
        POs (ndarray): array with 2 columns, each row is the (x, y)
            coordinates of a post office.
    Output:
        closest (ndarray): index of the closest post office to each resident,
            in terms of Euclidean distance.
    '''
    # Prepare a list of lists to store all distances
    distances = []
    # Loop over post offices
    for po in POs:
        dist_po = []
        # Loop over residents
        for res in residents:
            # Get the 2-norm of each vector between a resident and a PO
            dist_po.append(np.linalg.norm(res - po, ord=1))
        # Add the list of distances for all residents to that PO
        distances.append(dist_po)
    # Convert our list of lists to a NumPy array (transpose it)
    distances = np.array(distances).T
    # Return the index of the nearest PO, along each row of the array (find the min for each resident)
    closest = np.argmin(distances, axis=1)  #  np.argmin(): find the index of the smallest distance for each row
    return closest

# eg
def draw_PO_map(residents, POs):
    '''
    Display the nearest post office on the map for a population
    of residents, in different colours.

    Input:
        residents (ndarray): array with "pop" rows and 2 columns,
            each row is the (x, y) coordinates of 1 resident.
        POs (ndarray): array with n rows and 2 columns, each row
            is the (x, y) coordinates of 1 post office.

    Output: None
    '''
    # Get population size
    pop = residents.shape[0]
    # Set up a figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    # Make a colour map for the post offices
    colour_map = plt.get_cmap('gist_rainbow')
    colours = []
    n = POs.shape[0]
    for c in range(n):
        # Extract one colour, a fraction of the way into the colour map.
        # colour_map(0) gives us the leftmost colour in the 'gist_rainbow' map,
        # colour_map(1) gives us the rightmost colour.
        colours.append(colour_map(c/n))
    # Find the nearest PO for each resident, using the function we made earlier
    closest = nearest_PO(residents, POs)
    # Draw each cluster of residents with a different colour
    for po in range(n):
        # Boolean indexing: extract the rows (the residents) for whom "closest" is the current post office
        x = residents[closest == po, 0]
        y = residents[closest == po, 1]
        ax.plot(x, y, '.', color=colours[po], markersize=8)
    # Draw markers for each post office
    ax.plot(POs[:, 0], POs[:, 1], 'k*', markersize=15)
    plt.show()

# ! The moment of truth!
'''
Let's use all our functions to solve the task now -- finally!
'''
# eg

# Decide how many residents we want, and create them
pop = 10000
residents = make_residents(pop)
# Open our post offices across the city
n = 30
POs = create_POs(n)
# Now, draw the map!
draw_PO_map(residents, POs)

# Exercise 5
# We can redefine our function nearest_PO to also give us the distances
def nearest_PO(residents, POs, metric):
    '''
    Finds the closest post office to all residents.

    Input:
        residents (ndarray): array with "pop" rows and 2 columns,
            each row is the (x, y) coordinates of 1 resident.
        POs (ndarray): array with 2 columns, each row is the (x, y)
            coordinates of a post office.
        metric (str): 'manhattan' for Manhattan distance, 'euclid' for Euclidean distance.

    Output:
        closest (ndarray): index of the closest post office to each resident,
            in terms of Euclidean distance.
    '''
    if metric == 'manhattan':
        order = 1
    elif metric == 'euclid':
        order = 2
    else:
        print('Enter a valid metric!')
        return

    # Prepare a list of lists to store all distances
    distances = []

    # Loop over post offices
    for po in POs:
        dist_po = []
        # Loop over residents
        for res in residents:
            # Get the 2-norm of each vector between a resident and a PO
            dist_po.append(np.linalg.norm(res - po, ord=order))

        # Add the list of distances for all residents to that PO
        distances.append(dist_po)

    # Convert our list of lists to a NumPy array
    distances = np.array(distances).T

    # Return the index of the nearest PO, along each row of the array (find the min for each resident)
    closest = np.argmin(distances, axis=1)

    # Keep the distances
    pop = residents.shape[0]
    dist_closest = np.zeros(pop)
    for i in range(pop):
        dist_closest[i] = distances[i, closest[i]]

    return closest, dist_closest


def average_time(speed, pop):
    '''
    Plot average time to walk to the nearest PO as a function of number of POs.

    Input:
        speed (float): walking speed, km/h
        pop (int): population size for the simulation.
    Output: None
    '''
    av_time = np.zeros([30, 2])
    residents = make_residents(pop)

    # Loop over number of post offices
    for n in range(1, 31):
        POs = create_POs(n)

        # Get distances to nearest PO with both metrics
        _, dist1 = nearest_PO(residents, POs, 'manhattan')
        _, dist2 = nearest_PO(residents, POs, 'euclid')

        # Get average distance and time
        av_dist1 = np.mean(dist1)
        av_time[n-1, 0] = av_dist1 / speed
        av_dist2 = np.mean(dist2)
        av_time[n-1, 1] = av_dist2 / speed

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(range(1, 31), av_time[:, 0], '-', label='Manhattan distance')
    ax.plot(range(1, 31), av_time[:, 1], '-', label='Euclidean distance')
    ax.legend()
    ax.set_xlabel('Number of post offices')
    ax.set_ylabel('Average walking time (hrs)')
    plt.show()

# This takes a little while to run -- you can start with a smaller population
average_time(5, 1000)
